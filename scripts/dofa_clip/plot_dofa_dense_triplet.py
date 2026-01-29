import os, lmdb, pickle, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer


# ========= LoveDA class names (0..6) =========
CLASSES = ["background","building","road","water","barren land","forest","agricultural land"]
NUM_CLASSES = 7
IGNORE = 255

# ========= Palette (你现在用的 sam3 风格) =========
PALETTE_SAM3 = np.array([
    [128, 0, 128],   # 0 background
    [0, 255, 0],     # 1 building
    [0, 0, 0],       # 2 road
    [0, 0, 255],     # 3 water
    [128, 128, 0],   # 4 barren
    [0, 255, 255],   # 5 forest
    [255, 0, 255],   # 6 agriculture
], dtype=np.uint8)

def colorize_0_6_ignore255(label_hw: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = label_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)  # ignore 默认黑
    m = (label_hw != IGNORE)
    out[m] = palette[label_hw[m]]
    return out

def parse_vis_list(s: str):
    if s is None or s.strip() == "":
        return set()
    out = set()
    for t in s.split(","):
        t = t.strip()
        if t == "":
            continue
        out.add(int(t))
    return out

def safe_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def class_hist(label, num_classes=NUM_CLASSES, ignore=IGNORE):
    m = (label != ignore)
    if m.sum() == 0:
        return np.zeros((num_classes,), dtype=np.int64)
    v = label[m].astype(np.int64)
    v = v[(v >= 0) & (v < num_classes)]
    hist = np.bincount(v, minlength=num_classes)
    return hist

def save_triplet(img_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                 save_path: str, idx: int, palette: np.ndarray, extra_text: str = ""):
    im1 = Image.fromarray(img_rgb, mode="RGB")
    im2 = Image.fromarray(colorize_0_6_ignore255(gt, palette), mode="RGB")
    im3 = Image.fromarray(colorize_0_6_ignore255(pred, palette), mode="RGB")

    w, h = im1.size
    gap, top = 8, 64
    canvas = Image.new("RGB", (w * 3 + gap * 2, h + top), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = safe_font()

    draw.text((5, 5), f"Image (idx={idx})", fill=(0, 0, 0), font=font)
    draw.text((w + gap + 5, 5), "GT (color)", fill=(0, 0, 0), font=font)
    draw.text((w * 2 + gap * 2 + 5, 5), "Pred (color)", fill=(0, 0, 0), font=font)

    # 额外信息：unique / 占比等，帮助你判断“颜色问题还是预测问题”
    if extra_text:
        draw.text((5, 24), extra_text, fill=(0, 0, 0), font=font)

    canvas.paste(im1, (0, top))
    canvas.paste(im2, (w + gap, top))
    canvas.paste(im3, (w * 2 + gap * 2, top))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)

def update_conf(conf, gt, pred, num_classes=NUM_CLASSES, ignore=IGNORE):
    m = (gt != ignore) & (pred != ignore)
    if m.sum() == 0:
        return
    g = gt[m].astype(np.int64)
    p = pred[m].astype(np.int64)
    ok = (g >= 0) & (g < num_classes) & (p >= 0) & (p < num_classes)
    g = g[ok]; p = p[ok]
    # vectorized bincount over pairs
    idx = g * num_classes + p
    binc = np.bincount(idx, minlength=num_classes*num_classes)
    conf += binc.reshape(num_classes, num_classes)

def metrics_from_conf(conf):
    oa = np.trace(conf) / (conf.sum() + 1e-9)
    ious = []
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else np.nan)
    return float(oa), float(np.nanmean(ious)), ious


def main():
    ap = argparse.ArgumentParser()
    LMDB_PATH = "loveda_p512_val.lmdb"
    OUT_DIR   = "outs"
    ap.add_argument("--lmdb", default=LMDB_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)

    # 单张可视化
    ap.add_argument("--idx", type=int, default=0)

    # 批量计算 mIoU（只用画图脚本算，不改 eval）
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    ap.add_argument("--vis", type=str, default="", help="逗号分隔：比如 0,60,78 保存这些 triplet")

    ap.add_argument("--model_name", default="hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--stride", type=int, default=160)

    ap.add_argument("--tau", type=float, default=2.0, help="softmax temperature (>1 more uniform)")
    ap.add_argument("--topk", type=int, default=3, help="keep only top-k classes per pixel before normalize")
    ap.add_argument("--prompt_ensemble", action="store_true", help="use multiple prompts per class and average")

    args = ap.parse_args()

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model, preprocess = create_model_from_pretrained(args.model_name)
    tokenizer = get_tokenizer(args.model_name)
    model = model.to(device).eval()

    # prompts
    if args.prompt_ensemble:
        prompt_bank = {
            "background": ["remote sensing background", "other land cover", "unlabeled land cover"],
            "building": ["buildings from above", "houses in satellite imagery", "urban buildings"],
            "road": ["roads from above", "streets in satellite imagery", "paved road"],
            "water": ["river in satellite imagery", "lake in satellite imagery", "water body from above"],
            "barren land": ["bare soil", "barren ground", "exposed soil"],
            "forest": ["forest canopy", "woodland area", "trees from above"],
            "agricultural land": ["farmland", "crop fields", "agricultural fields from above"],
        }
        prompts = sum([prompt_bank[c] for c in CLASSES], [])
    else:
        prompts = [f"a remote sensing image of {c}" for c in CLASSES]

    text = tokenizer(prompts, context_length=model.context_length).to(device)
    with torch.no_grad():
        txt_all = F.normalize(model.encode_text(text), dim=-1)
        if args.prompt_ensemble:
            txt_feat = txt_all.view(NUM_CLASSES, -1, txt_all.shape[-1]).mean(1)
            txt_feat = F.normalize(txt_feat, dim=-1)
        else:
            txt_feat = txt_all

    # wvs
    wvs = torch.tensor([0.665, 0.560, 0.490], device=device)

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)

    vis_set = parse_vis_list(args.vis)
    palette = PALETTE_SAM3

    def run_one(idx, debug_first=False):
        key = str(idx).encode()
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            print("LMDB missing key:", idx)
            return None

        img_bytes, img_shape, lbl_bytes, lbl_shape = pickle.loads(raw)[:4]
        img_chw = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)   # BGR CHW
        lbl_hw  = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape)

        # 重要：确认 label 编码（0=ignore? 1..7=classes?）
        # 你当前逻辑是：lbl==0 -> ignore；其它 -1 -> 0..6
        if debug_first:
            u = np.unique(lbl_hw)
            print("DEBUG lbl_hw unique:", u[:50], "(len=", len(u), ")")

        img_rgb = img_chw[::-1].transpose(1,2,0)  # RGB HWC
        gt = np.where(lbl_hw == 0, IGNORE, lbl_hw - 1).astype(np.uint8)  # 0..6, ignore=255

        H, W = gt.shape
        logits_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
        count_map  = np.zeros((H, W), dtype=np.float32)

        CROP, STRIDE = args.crop, args.stride
        ys = list(range(0, max(H - CROP, 0) + 1, STRIDE))
        xs = list(range(0, max(W - CROP, 0) + 1, STRIDE))
        if len(ys) == 0: ys = [0]
        if len(xs) == 0: xs = [0]
        if ys[-1] != max(H - CROP, 0): ys.append(max(H - CROP, 0))
        if xs[-1] != max(W - CROP, 0): xs.append(max(W - CROP, 0))

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                crop = img_rgb[y:y+CROP, x:x+CROP, :]
                h0, w0 = crop.shape[:2]
                if h0 != CROP or w0 != CROP:
                    pad = np.zeros((CROP, CROP, 3), dtype=np.uint8)
                    pad[:h0, :w0, :] = crop
                    crop = pad

                im = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    tokens,_ = model.visual.trunk.forward_features(im, wvs)
                    N = tokens.shape[1]
                    g = int(np.sqrt(N))

                    # similarity
                    sim = torch.einsum("bnc,kc->bnk", F.normalize(tokens, dim=-1), txt_feat)
                    # sim = sim* model.logit_scale.exp() + model.logit_bias
                    sim = sim[0].permute(1, 0).reshape(NUM_CLASSES, g, g)
                    # sim = F.avg_pool2d(sim.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
                    sim = F.interpolate(sim.unsqueeze(0), size=(CROP, CROP),
                                        mode="bilinear", align_corners=False)[0]

                    prob = torch.softmax(sim, dim=0)  # (7,h0,w0)
                    crop_logits = prob.float().cpu().numpy()

                logits_sum[:, y:y+h0, x:x+w0] += crop_logits
                count_map[y:y+h0, x:x+w0] += 1.0

        count_map[count_map == 0] = 1.0
        logits_avg = logits_sum / count_map[None, :, :]
        pred = logits_avg.argmax(0).astype(np.uint8)
        pred[gt == IGNORE] = IGNORE

        return (img_rgb, gt, pred)

    # ========== 1) 批量算 mIoU ==========
    if args.start_idx is not None and args.end_idx is not None:
        s, e = int(args.start_idx), int(args.end_idx)
        assert e > s, "--end_idx must be > --start_idx"

        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        for idx in range(s, e):
            out = run_one(idx, debug_first=(idx == s))
            if out is None:
                continue
            img_rgb, gt, pred = out
            update_conf(conf, gt, pred)

            # 可选：保存指定 triplet
            if (idx in vis_set):
                gt_hist = class_hist(gt)
                pr_hist = class_hist(pred)
                gt_uni = np.where(gt_hist > 0)[0].tolist()
                pr_uni = np.where(pr_hist > 0)[0].tolist()
                extra = f"GT labels: {gt_uni} | Pred labels: {pr_uni}"
                save_path = os.path.join(args.out_dir, f"{idx:05d}_triplet.png")
                save_triplet(img_rgb, gt, pred, save_path, idx, palette, extra_text=extra)
                print("saved triplet:", save_path)

            if (idx - s + 1) % 10 == 0:
                oa, miou, _ = metrics_from_conf(conf)
                print(f"[{idx - s + 1:03d}/{e - s}] OA={oa:.4f} mIoU={miou:.4f}")

        oa, miou, ious = metrics_from_conf(conf)
        os.makedirs(args.out_dir, exist_ok=True)
        np.savez(os.path.join(args.out_dir, f"confmat_{s}_{e}.npz"), confmat=conf)

        print("=== DONE ===")
        print("range:", s, e)
        print("OA:", oa)
        print("mIoU:", miou)
        print("IoU per class:")
        for c, v in zip(CLASSES, ious):
            print(f"  {c:18s} {v if not np.isnan(v) else 'nan'}")
        print("saved confmat:", os.path.join(args.out_dir, f"confmat_{s}_{e}.npz"))
        return

    # ========== 2) 只画单张 ==========
    out = run_one(args.idx, debug_first=True)
    if out is None:
        raise RuntimeError(f"failed idx={args.idx}")

    img_rgb, gt, pred = out

    # 打印 unique labels 和占比（判断是不是 palette 错位/还是预测真错）
    gt_hist = class_hist(gt)
    pr_hist = class_hist(pred)
    gt_uni = np.where(gt_hist > 0)[0].tolist()
    pr_uni = np.where(pr_hist > 0)[0].tolist()

    gt_ratio = (gt_hist / max(gt_hist.sum(), 1)).round(3)
    pr_ratio = (pr_hist / max(pr_hist.sum(), 1)).round(3)

    print("DEBUG gt unique labels:", gt_uni)
    print("DEBUG pred unique labels:", pr_uni)
    print("DEBUG gt ratio:", gt_ratio.tolist())
    print("DEBUG pr ratio:", pr_ratio.tolist())

    extra = f"GT labels: {gt_uni} | Pred labels: {pr_uni}"
    save_path = os.path.join(args.out_dir, f"{args.idx:05d}_triplet.png")
    save_triplet(img_rgb, gt, pred, save_path, args.idx, PALETTE_SAM3, extra_text=extra)
    print("saved triplet:", save_path)


if __name__ == "__main__":
    main()
