import os, lmdb, pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

# ===== 固定路径：你按需改 =====
LMDB_PATH = "/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb"
OUT_DIR   = "/mnt/e/ovss_project/pythonProject1/runs/dofa_clip_loveda_val_0_200_dense"
os.makedirs(OUT_DIR, exist_ok=True)

# 0..6
CLASSES = ["background","building","road","water","barren land","forest","agricultural land"]
# 你也可以改成更“遥感”的模板
PROMPTS = [f"a remote sensing image of {c}" for c in CLASSES]

def metrics_from_conf(conf):
    oa = np.trace(conf) / (conf.sum() + 1e-9)
    ious = []
    for c in range(7):
        tp = conf[c,c]
        fp = conf[:,c].sum() - tp
        fn = conf[c,:].sum() - tp
        denom = tp + fp + fn
        ious.append(tp/denom if denom > 0 else np.nan)
    return float(oa), float(np.nanmean(ious)), ious

@torch.no_grad()
def try_get_tokens(model, im, wvs, device):
    """
    尝试从 vision trunk 拿到 token 特征 (B, N, C) 或 (B, C, H, W)。
    不同 open_clip / timm 版本结构差异很大，所以这里做多路 fallback。
    返回:
      tokens: torch.Tensor 或 None
      mode: 'tokens' / 'featmap' / 'global'
      global_feat: 如果拿不到 tokens，退化用 global embedding
    """
    trunk = model.visual.trunk

    # 1) timm ViT 常见：forward_features
    for call in [
        lambda: trunk.forward_features(im, wvs),
        lambda: trunk.forward_features(im),
    ]:
        try:
            out = call()
            if isinstance(out, (tuple, list)):
                # 取第一个像 token 的
                for x in out:
                    if torch.is_tensor(x) and x.ndim == 3:
                        return x, "tokens", None
                    if torch.is_tensor(x) and x.ndim == 4:
                        return x, "featmap", None
                # 否则看第一个
                out0 = out[0]
                if torch.is_tensor(out0) and out0.ndim == 3:
                    return out0, "tokens", None
                if torch.is_tensor(out0) and out0.ndim == 4:
                    return out0, "featmap", None
                if torch.is_tensor(out0) and out0.ndim == 2:
                    return None, "global", out0
            else:
                if torch.is_tensor(out) and out.ndim == 3:
                    return out, "tokens", None
                if torch.is_tensor(out) and out.ndim == 4:
                    return out, "featmap", None
                if torch.is_tensor(out) and out.ndim == 2:
                    return None, "global", out
        except Exception:
            pass

    # 2) 你旧脚本用 trunk(im, wvs) ——通常返回 global embedding
    try:
        out = trunk(im, wvs)
        if isinstance(out, (tuple, list)):
            # 找 tokens
            for x in out:
                if torch.is_tensor(x) and x.ndim == 3:
                    return x, "tokens", None
                if torch.is_tensor(x) and x.ndim == 4:
                    return x, "featmap", None
            out0 = out[0]
            if torch.is_tensor(out0) and out0.ndim == 2:
                return None, "global", out0
        else:
            if torch.is_tensor(out) and out.ndim == 2:
                return None, "global", out
    except Exception:
        pass

    return None, "global", None

def save_triplet(img_rgb, gt, pred, save_path):
    """
    img_rgb: HWC uint8
    gt/pred: HW uint8 with 255 ignore, class 0..6
    """
    import matplotlib.pyplot as plt

    def colorize(label):
        # 简单调色板（固定 7 类 + ignore=黑）
        palette = np.array([
            [0, 255, 0],    # 0 background
            [0, 128, 255],  # 1 building
            [255, 0, 255],  # 2 road
            [0, 0, 255],    # 3 water
            [255, 255, 0],  # 4 barren
            [0, 128, 0],    # 5 forest
            [128, 0, 128],  # 6 agriculture
        ], dtype=np.uint8)
        out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        m = (label != 255)
        out[m] = palette[label[m]]
        return out

    gt_c  = colorize(gt)
    pr_c  = colorize(pred)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Image"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(gt_c);   plt.title("GT");    plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(pr_c);   plt.title("Pred");  plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Model =====
    model_name = "hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    model = model.to(device).eval()

    # text features
    text = tokenizer(PROMPTS, context_length=model.context_length).to(device)
    with torch.no_grad():
        txt_feat = F.normalize(model.encode_text(text), dim=-1)  # (7, D)

    # 旧脚本里固定的 wvs
    wvs = torch.tensor([0.665, 0.560, 0.490], device=device)

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
    conf_total = np.zeros((7,7), dtype=np.int64)

    # dense sliding params
    CROP = 384
    STRIDE = 192
    NUM = 200

    # 可视化：输出第 0 张 triplet
    VIS_INDEX = 0

    for idx in range(NUM):
        key = str(idx).encode()
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            print("LMDB missing key:", idx)
            continue

        img_bytes, img_shape, lbl_bytes, lbl_shape = pickle.loads(raw)[:4]
        img_chw = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)   # BGR CHW
        lbl_hw  = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape)

        img_rgb = img_chw[::-1].transpose(1,2,0)  # RGB HWC
        gt = np.where(lbl_hw == 0, 255, lbl_hw - 1).astype(np.uint8)

        H, W = gt.shape
        # logits accumulator: (7, H, W)
        logits_sum = np.zeros((7, H, W), dtype=np.float32)
        count_map  = np.zeros((H, W), dtype=np.float32)

        # sliding windows
        ys = list(range(0, max(H - CROP, 0) + 1, STRIDE))
        xs = list(range(0, max(W - CROP, 0) + 1, STRIDE))
        if len(ys) == 0: ys = [0]
        if len(xs) == 0: xs = [0]
        if ys[-1] != max(H - CROP, 0): ys.append(max(H - CROP, 0))
        if xs[-1] != max(W - CROP, 0): xs.append(max(W - CROP, 0))

        for y in ys:
            for x in xs:
                crop = img_rgb[y:y+CROP, x:x+CROP, :]
                h0, w0 = crop.shape[:2]

                # pad to CROP
                if h0 != CROP or w0 != CROP:
                    pad = np.zeros((CROP, CROP, 3), dtype=np.uint8)
                    pad[:h0, :w0, :] = crop
                    crop = pad

                im = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    tokens, mode, global_feat = try_get_tokens(model, im, wvs, device)

                    if mode == "tokens" and tokens is not None:
                        # tokens: (B, N, C) 可能含 cls token
                        tok = tokens
                        if tok.shape[1] > 1:
                            tok2 = tok[:, 1:, :]  # drop cls
                        else:
                            tok2 = tok

                        # 推断 grid
                        n = tok2.shape[1]
                        g = int(np.sqrt(n))
                        if g * g != n:
                            # 拿不到规整 grid，就退化成 global
                            mode = "global"
                        else:
                            # (1, n, C) @ (C, 7) -> (1, n, 7)
                            sim = torch.einsum("bnc,kc->bnk", F.normalize(tok2, dim=-1), txt_feat)
                            sim = sim[0].permute(1,0).reshape(7, g, g)  # (7,g,g)
                            sim = F.interpolate(sim.unsqueeze(0), size=(CROP, CROP),
                                                mode="bilinear", align_corners=False)[0]
                            crop_logits = sim[:, :h0, :w0].float().cpu().numpy()

                    if mode != "tokens":
                        # fallback: 只拿 global embedding，对整个 crop 赋同一类（很粗，但能跑通）
                        if global_feat is None:
                            out = model.visual.trunk(im, wvs)
                            global_feat = out[0] if isinstance(out, (tuple, list)) else out
                        img_feat = F.normalize(global_feat, dim=-1)  # (1, D)
                        sim = (img_feat @ txt_feat.T)[0]  # (7,)
                        cls = int(sim.argmax().item())
                        crop_logits = np.zeros((7, h0, w0), dtype=np.float32)
                        crop_logits[cls, :, :] = 1.0

                logits_sum[:, y:y+h0, x:x+w0] += crop_logits
                count_map[y:y+h0, x:x+w0] += 1.0

        # average + predict
        count_map[count_map == 0] = 1.0
        logits_avg = logits_sum / count_map[None, :, :]
        pred = logits_avg.argmax(0).astype(np.uint8)
        pred[gt == 255] = 255

        # eval
        valid = (gt != 255) & (pred != 255)
        gtv = gt[valid].astype(np.int64)
        prv = pred[valid].astype(np.int64)
        for g, p in zip(gtv, prv):
            if 0 <= g < 7 and 0 <= p < 7:
                conf_total[g,p] += 1

        # save one triplet
        if idx == VIS_INDEX:
            save_path = os.path.join(OUT_DIR, f"{idx:05d}_triplet.png")
            save_triplet(img_rgb, gt, pred, save_path)
            print("saved triplet:", save_path)

        if (idx+1) % 10 == 0:
            oa, miou, _ = metrics_from_conf(conf_total)
            print(f"[{idx+1:03d}/{NUM}] OA={oa:.4f} mIoU={miou:.4f}")

    oa, miou, ious = metrics_from_conf(conf_total)
    np.savez(os.path.join(OUT_DIR, "confmat.npz"), confmat=conf_total)

    print("=== DONE 0-199 (DENSE) ===")
    print("OA:", oa)
    print("mIoU:", miou)
    print("IoU per class:")
    for c, v in zip(CLASSES, ious):
        print(f"  {c:18s} {v if not np.isnan(v) else 'nan'}")
    print("saved:", os.path.join(OUT_DIR, "confmat.npz"))

if __name__ == "__main__":
    main()
