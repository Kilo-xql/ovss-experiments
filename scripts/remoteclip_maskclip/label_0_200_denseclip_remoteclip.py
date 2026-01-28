import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip


# ======== LoveDA 7 classes (labels 1..7), ignore=0 ========
# label id -> class:
# 1 background, 2 building, 3 road, 4 water, 5 barren, 6 forest, 7 agriculture

CLASS_PHRASES = {
    "background": [
        "other terrain",
        "unknown region",
        "unlabeled area",
        "miscellaneous land cover",
        "background region",
    ],
    "buildings": [
        "buildings",
        "residential buildings",
        "urban area",
        "dense houses",
        "industrial buildings",
        "man-made structures",
    ],
    "roads": [
        "road",
        "asphalt road",
        "concrete road",
        "paved road",
        "street",
        "highway",
        "expressway",
        "road surface",
        "road network",
        "road intersection",
        "impervious surface",
        "urban impervious surface",
        "gray impervious surface",
        "linear road",
    ],
    "water": [
        "water body",
        "river",
        "lake",
        "pond",
        "reservoir",
        "water surface",
    ],
    "barren land": [
        "bare soil",
        "bare land",
        "barren ground",
        "dry soil",
        "exposed earth",
        "sand or soil",
    ],
    "forest": [
        "forest",
        "woodland",
        "dense trees",
        "green vegetation",
        "tree canopy",
        "tree cover",
    ],
    "agriculture": [
        "farmland",
        "cropland",
        "cultivated field",
        "crop rows",
        "agricultural land",
        "planted field",
    ],
}

CLASS_SIMPLE = {
    "background": ["background"],
    "buildings": ["building", "buildings", "urban area"],
    "roads": ["road", "roads"],
    "water": ["water", "river", "lake"],
    "barren land": ["barren land", "bare land"],
    "forest": ["forest", "trees"],
    "agriculture": ["farmland", "cropland", "agriculture"],
}

TEMPLATES = [
    "a remote sensing photo of {}",
    "a satellite image of {}",
    "an aerial photo of {}",
    "an overhead view of {}",
    "a remote sensing scene of {}",
]

# ===== debug switches =====
_DENSE_DEBUG_DONE = False

CLASS_ORDER = [
    "background",
    "buildings",
    "roads",
    "water",
    "barren land",
    "forest",
    "agriculture",
]



def fast_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_label: int = 0):
    mask = (y_true != ignore_label)
    yt = y_true[mask].astype(np.int64)
    yp = y_pred[mask].astype(np.int64)

    valid = (yt >= 1) & (yt <= num_classes) & (yp >= 1) & (yp <= num_classes)
    yt = yt[valid] - 1
    yp = yp[valid] - 1

    k = num_classes
    return np.bincount(k * yt + yp, minlength=k * k).reshape(k, k)


def miou_from_cm(cm: np.ndarray):
    diag = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag
    iou = np.divide(diag, union, out=np.zeros_like(diag), where=(union > 0))
    return float(np.mean(iou)), iou


def _to_rgb_np(img_np: np.ndarray) -> np.ndarray:
    """
    v2 对齐版：
    - CHW(3,H,W): BGR->RGB + 转 HWC
    - HWC(H,W,3): 也按 BGR->RGB swap（很多 LMDB 保存流会这样）
    """
    img_np = np.asarray(img_np).astype(np.uint8)

    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = img_np[[2, 1, 0], :, :]
        img_np = np.transpose(img_np, (1, 2, 0))
        return img_np

    if img_np.ndim == 3 and img_np.shape[2] == 3:
        return img_np[:, :, [2, 1, 0]]

    if img_np.ndim == 2:
        return np.stack([img_np, img_np, img_np], axis=-1)

    raise ValueError(f"Unsupported image shape: {img_np.shape}")


def read_lmdb_item(txn, idx: int):
    raw = txn.get(str(idx).encode())
    if raw is None:
        return None
    obj = pickle.loads(raw)
    img = np.frombuffer(obj[0], dtype=np.uint8).reshape(obj[1]).copy()
    lbl = np.frombuffer(obj[2], dtype=np.uint8).reshape(obj[3]).copy()
    return img, lbl


def _extract_normalize_from_preprocess(preprocess):
    mean = None
    std = None
    if hasattr(preprocess, "transforms"):
        for t in preprocess.transforms:
            if t.__class__.__name__.lower() == "normalize":
                mean = getattr(t, "mean", None)
                std = getattr(t, "std", None)
                break
    if mean is None or std is None:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    return mean, std


def preprocess_dense_keep_size(pil_img: Image.Image, mean, std) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
    mean_t = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    t = (t - mean_t) / std_t
    return t


# ================= Text features（严格按学姐要求）=================

def build_text_features(model, tokenizer, device, phrases_dict):
    """
    规则（严格按学姐）：
      - 同一 phrase：只在 templates 之间平均
      - 同一类多个 phrases：不平均！全部保留
      - 后面与 image 相似度：同类取 max
    return:
      text_feat: (7, Pmax, D) normalized
      mask:      (7, Pmax) bool
    """
    class_order = [
        "background",
        "buildings",
        "roads",
        "water",
        "barren land",
        "forest",
        "agriculture",
    ]

    num_phrases = [len(phrases_dict[c]) for c in class_order]
    if min(num_phrases) <= 0:
        raise ValueError("Each class must have at least 1 phrase.")
    Pmax = max(num_phrases)

    with torch.no_grad():
        tmp_prompt = TEMPLATES[0].format(phrases_dict[class_order[0]][0])
        tmp_tok = tokenizer([tmp_prompt]).to(device)
        D = model.encode_text(tmp_tok).shape[-1]

        text_feat = torch.zeros((len(class_order), Pmax, D), device=device, dtype=torch.float32)
        mask = torch.zeros((len(class_order), Pmax), device=device, dtype=torch.bool)

        for ci, cname in enumerate(class_order):
            for pi, ph in enumerate(phrases_dict[cname]):
                prompts = [tmpl.format(ph) for tmpl in TEMPLATES]
                tokens = tokenizer(prompts).to(device)
                tf = model.encode_text(tokens)          # (T, D)
                tf = F.normalize(tf, dim=-1)

                ph_tf = tf.mean(dim=0)                  # ✅ 只平均 templates
                ph_tf = F.normalize(ph_tf, dim=-1)

                text_feat[ci, pi] = ph_tf
                mask[ci, pi] = True

    return text_feat, mask


# ================= DenseCLIP core =================

def _get_patch_size(model):
    vis = getattr(model, "visual", None)
    for path in [
        ("patch_size",),
        ("trunk", "patch_embed", "patch_size"),
        ("patch_embed", "patch_size"),
    ]:
        obj = vis
        ok = True
        for k in path:
            if obj is None or not hasattr(obj, k):
                ok = False
                break
            obj = getattr(obj, k)
        if not ok:
            continue
        if isinstance(obj, (tuple, list)):
            return int(obj[0])
        if isinstance(obj, int):
            return int(obj)

    if hasattr(vis, "conv1") and hasattr(vis.conv1, "kernel_size"):
        ks = vis.conv1.kernel_size
        return int(ks[0]) if isinstance(ks, (tuple, list)) else int(ks)

    raise RuntimeError("Cannot infer patch size from model.visual.*")


def _pad_to_multiple(img_tensor, multiple: int):
    _, _, H, W = img_tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img_tensor, (H, W)
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return img_tensor, (H, W)


@torch.no_grad()
def _get_vit_tokens_fallback(model, img_tensor):
    """
    这是你原来那套手写 token 提取（保留作为 fallback）。
    """
    vis = model.visual

    if all(hasattr(vis, k) for k in ["conv1", "transformer"]):
        x = vis.conv1(img_tensor)  # (B, D, Gh, Gw)
        B, D, Gh, Gw = x.shape
        x = x.reshape(B, D, Gh * Gw).permute(0, 2, 1)  # (B, N, D)

        if hasattr(vis, "class_embedding"):
            cls = vis.class_embedding.to(x.dtype).view(1, 1, D).expand(B, 1, D)
            x = torch.cat([cls, x], dim=1)
        elif hasattr(vis, "cls_token"):
            cls = vis.cls_token.to(x.dtype).expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        if hasattr(vis, "positional_embedding"):
            pe = vis.positional_embedding.to(x.dtype)  # (1+N0, D)
            if pe.dim() == 2 and pe.shape[0] != x.shape[1]:
                pe_cls = pe[:1, :]
                pe_grid = pe[1:, :]
                gs = int(pe_grid.shape[0] ** 0.5)
                pe_grid = pe_grid.reshape(gs, gs, -1).permute(2, 0, 1).unsqueeze(0)
                pe_grid = F.interpolate(pe_grid, size=(Gh, Gw), mode="bicubic", align_corners=False)
                pe_grid = pe_grid.squeeze(0).permute(1, 2, 0).reshape(Gh * Gw, -1)
                pe = torch.cat([pe_cls, pe_grid], dim=0)
            x = x + pe.unsqueeze(0) if pe.dim() == 2 else (x + pe)

        if hasattr(vis, "ln_pre"):
            x = vis.ln_pre(x)
        elif hasattr(vis, "norm_pre"):
            x = vis.norm_pre(x)

        xt = x.permute(1, 0, 2)
        xt = vis.transformer(xt)
        if isinstance(xt, (tuple, list)):
            xt = xt[0]
        x = xt.permute(1, 0, 2)

        if hasattr(vis, "ln_post"):
            x = vis.ln_post(x)
        elif hasattr(vis, "norm"):
            x = vis.norm(x)

        return x

    raise RuntimeError("visual encoder is not CLIP ViT-like (missing conv1/transformer).")


@torch.no_grad()
def get_vit_tokens_interp(model, img_tensor, return_pre_ln_post: bool = False):
    """
    支持任意 H,W (只要能被 patch_size 整除，或至少 conv1 能得到网格) 的 ViT token 抽取：
      - 手动：conv1 -> flatten -> concat cls -> pos_embed(网格部分bicubic插值) -> ln_pre
      - transformer：支持 resblocks/blocks，并兼容 block 输入 (B,L,D) / (L,B,D)
      - return_pre_ln_post=True: 返回 transformer 输出（未 ln_post）
      - 否则：返回 ln_post 后 tokens

    额外：
      - 允许通过 get_vit_tokens_interp._take_layer = k 截断到第 k 层（0-based）
    """
    vis = model.visual

    # --- 必要组件检查：缺了就退回你已有的 fallback（224 严格那套/或你手写那套） ---
    need = ["conv1", "class_embedding", "positional_embedding"]
    if not all(hasattr(vis, k) for k in need):
        # 这里用你文件里已有的 fallback（它是 CLIP 原版路径）
        return _get_vit_tokens_fallback(model, img_tensor)

    # --- 1) conv -> patch tokens ---
    x = vis.conv1(img_tensor)                    # (B, D, Gh, Gw)
    B, D, Gh, Gw = x.shape
    x = x.reshape(B, D, Gh * Gw).permute(0, 2, 1)  # (B, N, D)

    # --- 2) concat cls token ---
    cls = vis.class_embedding.to(x.dtype).view(1, 1, D).expand(B, 1, D)
    x = torch.cat([cls, x], dim=1)               # (B, 1+N, D)

    # --- 3) positional embedding：网格部分插值到 (Gh, Gw) ---
    pe = vis.positional_embedding.to(x.dtype)    # (1+N0, D) 或 (1,1+N0,D) 不同版本
    if pe.dim() == 3:
        pe = pe.squeeze(0)                       # -> (1+N0, D)

    # pe 期望：1 + gs0*gs0
    if pe.shape[0] != x.shape[1]:
        pe_cls = pe[:1, :]                       # (1, D)
        pe_grid = pe[1:, :]                      # (N0, D)

        gs0 = int(pe_grid.shape[0] ** 0.5)
        if gs0 * gs0 != pe_grid.shape[0]:
            # 很少见：pos_embed 不是正方形网格，直接退回不插值的安全模式
            #（至少不崩）
            pe = pe[: x.shape[1], :]
        else:
            pe_grid = pe_grid.reshape(gs0, gs0, -1).permute(2, 0, 1).unsqueeze(0)  # (1,D,gs0,gs0)
            pe_grid = F.interpolate(pe_grid, size=(Gh, Gw), mode="bicubic", align_corners=False)
            pe_grid = pe_grid.squeeze(0).permute(1, 2, 0).reshape(Gh * Gw, -1)     # (N,D)
            pe = torch.cat([pe_cls, pe_grid], dim=0)                               # (1+N, D)

    x = x + pe.unsqueeze(0)                   # (B, 1+N, D)

    # --- 4) ln_pre（如果有）---
    if hasattr(vis, "ln_pre"):
        x = vis.ln_pre(x)
    elif hasattr(vis, "norm_pre"):
        x = vis.norm_pre(x)

    # --- 5) transformer：优先逐 block 跑，兼容 (B,L,D)/(L,B,D)，支持截断 ---
    take_layer = getattr(get_vit_tokens_interp, "_take_layer", None)  # None=跑完整个 transformer

    blocks = None
    if hasattr(vis.transformer, "resblocks"):
        blocks = vis.transformer.resblocks
    elif hasattr(vis.transformer, "blocks"):
        blocks = vis.transformer.blocks

    def _run_block(block, x_bl):
        # 兼容 block 期望 (L,B,D) 的情况
        try:
            return block(x_bl)
        except Exception:
            y = block(x_bl.permute(1, 0, 2))
            return y.permute(1, 0, 2)

    if blocks is not None:
        for li, blk in enumerate(blocks):
            x = _run_block(blk, x)
            if take_layer is not None and li == take_layer:
                break
    else:
        # 兜底：直接调用 transformer（同样做一次维度兼容）
        try:
            x = vis.transformer(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
        except Exception:
            xt = vis.transformer(x.permute(1, 0, 2))
            if isinstance(xt, (tuple, list)):
                xt = xt[0]
            x = xt.permute(1, 0, 2)

    # --- 6) ln_post / norm ---
    if return_pre_ln_post:
        return x

    if hasattr(vis, "ln_post"):
        x = vis.ln_post(x)
    elif hasattr(vis, "norm"):
        x = vis.norm(x)

    return x


def _apply_vision_proj(model, x: torch.Tensor) -> torch.Tensor:
    vis = model.visual
    if hasattr(vis, "proj"):
        p = vis.proj
        if isinstance(p, torch.nn.Linear):
            return p(x)
        if torch.is_tensor(p) and p.dim() == 2:
            if p.shape[0] == x.shape[-1]:
                return x @ p
            if p.shape[1] == x.shape[-1]:
                return x @ p.t()

    if hasattr(model, "visual_projection"):
        p = model.visual_projection
        if isinstance(p, torch.nn.Linear):
            return p(x)
        if torch.is_tensor(p) and p.dim() == 2:
            if p.shape[0] == x.shape[-1]:
                return x @ p
            if p.shape[1] == x.shape[-1]:
                return x @ p.t()

    raise RuntimeError("Cannot find valid vision projection to map tokens dim -> text dim.")

@torch.no_grad()
def dense_logits_crop(
    model,
    crop_bchw: torch.Tensor,      # (1,3,Hc,Wc)  Hc=Wc=crop_size (224/448)
    txt_feat_7pd: torch.Tensor,   # (7,P,D)
    txt_mask_7p: torch.Tensor,    # (7,P)
    device: str,
    debug: bool = False,
):
    crop = crop_bchw.to(device)

    # post 用于 debug cls 对齐；pre 用于 patch dense
    tokens_post = get_vit_tokens_interp(model, crop, return_pre_ln_post=False)  # ln_post后
    tokens_pre  = get_vit_tokens_interp(model, crop, return_pre_ln_post=True)   # ln_post前

    tokens_all = tokens_post
    patch_tok = tokens_pre[:, 1:, :]

    # ===== debug（可选）=====
    global _DENSE_DEBUG_DONE
    if debug and (not _DENSE_DEBUG_DONE):
        _DENSE_DEBUG_DONE = True
        print("[DBG] tokens_all.shape =", tuple(tokens_all.shape),
              "patch_tok.shape =", tuple(patch_tok.shape),
              "txt_feat.shape =", tuple(txt_feat_7pd.shape))

    # 投影到 text dim（如果需要）
    if patch_tok.shape[-1] != txt_feat_7pd.shape[-1]:
        patch_tok = _apply_vision_proj(model, patch_tok)

    patch_tok = F.normalize(patch_tok, dim=-1)

    # sim: (1, N, 7, P)
    sim = torch.einsum("bnd,cpd->bncp", patch_tok, txt_feat_7pd)
    sim = sim.masked_fill(~txt_mask_7p[None, None, :, :].to(device), -1e9)

    # 类内 max： (1, N, 7)
    logits = sim.max(dim=-1).values

    # 动态网格 reshape：N = gh*gw
    ps = 32  # ViT-B-32
    gh = crop.shape[-2] // ps
    gw = crop.shape[-1] // ps
    logits = logits.permute(0, 2, 1).reshape(1, 7, gh, gw)   # (1,7,gh,gw)

    # ✅ 上采样回 crop 像素大小，保证能累加到 preds
    logits = F.interpolate(logits, size=crop.shape[-2:], mode="bilinear", align_corners=False)  # (1,7,Hc,Wc)

    return logits


@torch.no_grad()
def forward_slide_logits(
    model,
    img_bchw: torch.Tensor,       # (1,3,H,W) 这里 H=W=512
    txt_feat_7pd: torch.Tensor,
    txt_mask_7p: torch.Tensor,
    device: str,
    stride: int = 112,
    crop_size: int = 224,
):
    img = img_bchw.to(device)
    B, C, H, W = img.shape
    assert B == 1, "this script assumes batch=1"

    preds = img.new_zeros((1, 7, H, W))
    count = img.new_zeros((1, 1, H, W))

    h_grids = max(H - crop_size + stride - 1, 0) // stride + 1
    w_grids = max(W - crop_size + stride - 1, 0) // stride + 1

    for hi in range(h_grids):
        for wi in range(w_grids):
            y1 = hi * stride
            x1 = wi * stride
            y2 = min(y1 + crop_size, H)
            x2 = min(x1 + crop_size, W)
            y1 = max(y2 - crop_size, 0)
            x1 = max(x2 - crop_size, 0)

            crop = img[:, :, y1:y2, x1:x2]  # (1,3,224,224) at borders too

            # 如果边缘不足 224，pad 到 224（SegEarth 也做 padding）
            pad_h = crop_size - crop.shape[-2]
            pad_w = crop_size - crop.shape[-1]
            if pad_h > 0 or pad_w > 0:
                crop = F.pad(crop, (0, pad_w, 0, pad_h), mode="constant", value=0)

            crop_logits = dense_logits_crop(
                model, crop, txt_feat_7pd, txt_mask_7p, device=device,
                debug=getattr(forward_slide_logits, "_debug", False)
            )


            # 去掉 padding 区域
            crop_logits = crop_logits[:, :, : (y2 - y1), : (x2 - x1)]

            preds[:, :, y1:y2, x1:x2] += crop_logits
            count[:, :, y1:y2, x1:x2] += 1

    preds = preds / count.clamp_min(1.0)
    return preds  # (1,7,H,W)


@torch.no_grad()
def predict_dense_tta(
    model,
    img_tensor_bchw: torch.Tensor,
    txt_feat_7pd: torch.Tensor,
    txt_mask_7p: torch.Tensor,
    device: str,
    scales=(1.0,),
    hflip: bool = True,
    slide_stride: int = 112,
    slide_crop: int = 224,
):
    img = img_tensor_bchw.to(device)
    H, W = img.shape[-2], img.shape[-1]

    logits_acc = None
    count = 0

    for s in scales:
        if abs(s - 1.0) < 1e-6:
            img_s = img
        else:
            img_s = F.interpolate(
                img, size=(int(H * s), int(W * s)),
                mode="bilinear", align_corners=False
            )

        forward_slide_logits._debug = getattr(predict_dense_tta, "_debug", False)

        lg = forward_slide_logits(
            model, img_s, txt_feat_7pd, txt_mask_7p, device=device,
            stride=slide_stride, crop_size=slide_crop
        )
        lg = F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)

        logits_acc = lg if logits_acc is None else (logits_acc + lg)
        count += 1

        if hflip:
            img_f = torch.flip(img_s, dims=[-1])
            lgf = forward_slide_logits(
                model, img_f, txt_feat_7pd, txt_mask_7p, device=device,
                stride=slide_stride, crop_size=slide_crop
            )
            lgf = torch.flip(lgf, dims=[-1])
            lgf = F.interpolate(lgf, size=(H, W), mode="bilinear", align_corners=False)

            logits_acc = logits_acc + lgf
            count += 1

    # ✅ 一定要在循环外做平均
    logits_full = logits_acc / float(count)   # (1,7,H,W)

    # ---- temperature ----
    tau = getattr(predict_dense_tta, "_tau", 1.0)
    logits_full = logits_full / float(tau)

    # ---- fg-first decoding ----
    fg_logits = logits_full[:, 1:, :, :]          # (1,6,H,W)
    fg_val, fg_idx = fg_logits.max(dim=1)         # (1,H,W), idx in [0..5]
    pred = (fg_idx + 2).squeeze(0).to(torch.uint8).cpu()  # -> label [2..7]

    # ---- margin back to bg ----
    bg_logit = logits_full[:, 0, :, :]            # (1,H,W)
    score = (fg_val - bg_logit).squeeze(0).cpu()  # (H,W)
    m = float(getattr(predict_dense_tta, "_fg_margin", 0.0))

    # ===== DEBUG：只在 debug_first 的第一张图打印一次 score 尺度 =====
    if getattr(predict_dense_tta, "_debug", False):
        sc = score
        print("[DBG] score(fg-bg) stats:",
            "min", float(sc.min()),
            "max", float(sc.max()),
            "mean", float(sc.mean()))
        # 看看不同阈值会把多少像素打回背景
        for th in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]:
            frac = float((sc < th).float().mean())
            print(f"[DBG] frac(score<{th}) = {frac:.4f}")

    pred[score < m] = 1


    return pred



@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--save_pred", action="store_true")
    ap.add_argument("--debug_first", action="store_true")
    ap.add_argument("--no_hflip", action="store_true")

    # 兼容旧命令：保留但不再使用（学姐要求去掉偏置处理）
    ap.add_argument("--bias_fix", action="store_true", help="(ignored) bias ops are disabled per advisor.")

    ap.add_argument("--text_set", type=str, default="simple", choices=["simple", "phrase"])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])

    ap.add_argument("--slide_crop", type=int, default=224)
    ap.add_argument("--slide_stride", type=int, default=112)

    ap.add_argument("--tau", type=float, default=1.0, help="softmax temperature on logits (smaller=sharper)")
    ap.add_argument("--fg_margin", type=float, default=0.0, help="foreground-vs-background margin threshold")



    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "denseclip_pred_png"
    if args.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 按 v2 的方式加载 RemoteCLIP 权重
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.ckpt)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # ---- sanity checks ----
    with torch.no_grad():
        if hasattr(model, "logit_scale"):
            print("[CHK] logit_scale.exp() =", float(model.logit_scale.exp().cpu().item()))
        w = getattr(getattr(model.visual, "conv1", None), "weight", None)
        if w is not None:
            print("[CHK] visual.conv1.weight mean/std =",
                  float(w.mean().cpu().item()), float(w.std().cpu().item()))

    # SegEarth-OV 的归一化
    mean, std = _extract_normalize_from_preprocess(preprocess)

    # ---- text features ----
    text_dict = CLASS_SIMPLE if args.text_set == "simple" else CLASS_PHRASES
    text_feat, text_mask = build_text_features(model, tokenizer, device, text_dict)
    text_feat = text_feat.to(device)
    text_mask = text_mask.to(device)

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    cm = np.zeros((7, 7), dtype=np.int64)

    with env.begin() as txn:
        for j, idx in enumerate(range(args.start, args.end)):
            item = read_lmdb_item(txn, idx)
            if item is None:
                print(f"[WARN] missing idx={idx}, skip")
                continue

            img_raw, gt = item
            img_rgb = _to_rgb_np(img_raw)
            pil_img = Image.fromarray(img_rgb, mode="RGB")

            inp = preprocess_dense_keep_size(pil_img, mean, std).unsqueeze(0)

            debug = args.debug_first and (idx == args.start)
            if debug:
                print("[DBG] gt unique:", np.unique(gt))
                print("[DBG] inp shape:", tuple(inp.shape))

            predict_dense_tta._debug = debug
            predict_dense_tta._tau = args.tau
            predict_dense_tta._fg_margin = args.fg_margin

            pred = predict_dense_tta(
                model=model,
                img_tensor_bchw=inp,
                txt_feat_7pd=text_feat,
                txt_mask_7p=text_mask,
                device=device,
                scales=tuple(args.scales),
                hflip=(not args.no_hflip),
                slide_stride=args.slide_stride,
                slide_crop=args.slide_crop,
            )


            cm += fast_confusion_matrix(gt, pred.numpy().astype(np.uint8), num_classes=7, ignore_label=0)

            if args.save_pred and (idx - args.start) < 50:
                Image.fromarray(pred.numpy()).save(pred_dir / f"{idx:05d}.png")

            if (j + 1) % 20 == 0:
                miou, ious = miou_from_cm(cm)
                print(f"[{idx}] mIoU={miou:.4f} per-class={np.round(ious, 4)}")

    miou, ious = miou_from_cm(cm)
    print("\n=== FINAL (DenseCLIP: template-avg, phrase-max; NO bias/centering) ===")
    print("mIoU:", miou)
    print("IoU per class:", ious)

    labels = np.array(["background", "building", "road", "water", "barren", "forest", "agriculture"], dtype=object)
    np.savez_compressed(out_dir / "confmat_denseclip.npz", confmat=cm, labels=labels)
    print("Saved:", str(out_dir / "confmat_denseclip.npz"))
    print("[DBG] TTA scales:", args.scales, "hflip:", (not args.no_hflip))


if __name__ == "__main__":
    main()
