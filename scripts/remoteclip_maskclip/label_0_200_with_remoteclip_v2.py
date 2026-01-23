import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image
import torch
import open_clip


# ========= LoveDA 7 classes (output 1..7) =========
DEFAULT_AGRI_TEXT = "cultivated farmland with crop rows"

CLASS_NAMES = [
    "background",
    "buildings",
    "roads",
    "water",
    "barren land",
    "forest",
    DEFAULT_AGRI_TEXT,  # agriculture
]

TEMPLATES = [
    "a remote sensing photo of {}",
    "a satellite image of {}",
    "an aerial photo of {}",
    "an overhead view of {}",
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
    """LMDB image usually CHW (3,H,W) BGR; convert to HWC RGB uint8."""
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


def mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def pad_box(box, H, W, pad):
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def build_text_features(model, tokenizer, device, class_names):
    prompts = []
    for tmpl in TEMPLATES:
        for cname in class_names:
            prompts.append(tmpl.format(cname))

    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        tf = model.encode_text(tokens)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    T = len(TEMPLATES)
    C = len(class_names)
    tf = tf.view(T, C, -1).mean(dim=0)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf  # [C, D]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--npz_dir", required=True, help="dir containing 00000.npz..")
    ap.add_argument("--ckpt", required=True, help="RemoteCLIP .pt")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)

    # ✅ 最小影响：固定默认 pad=16（比 32 更稳）
    ap.add_argument("--pad", type=int, default=16)

    ap.add_argument("--min_area", type=int, default=64)
    ap.add_argument("--max_masks", type=int, default=60)
    ap.add_argument("--score_gamma", type=float, default=1.0)
    ap.add_argument("--save_pred", action="store_true")

    # ✅ 最小提升：农业通道只乘一次（别重复乘）
    ap.add_argument("--agri_boost", type=float, default=1.4)

    # ✅ 允许你不改代码直接试农业 prompt
    ap.add_argument("--agri_text", type=str, default=DEFAULT_AGRI_TEXT)

    # ✅ 可选：把 sims 平移成非负，避免 background=0 把负分压死（默认关，不影响原行为）
    ap.add_argument("--force_pos", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "maskclip_pred_png"
    if args.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.ckpt)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    class_names = CLASS_NAMES.copy()
    class_names[6] = args.agri_text
    text_feat = build_text_features(model, tokenizer, device, class_names)  # [7, D]

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
            H, W = gt.shape

            npz_path = Path(args.npz_dir) / f"{idx:05d}.npz"
            if not npz_path.exists():
                print(f"[WARN] missing proposals {npz_path}, skip idx={idx}")
                continue

            z = np.load(npz_path, allow_pickle=True)
            masks = z["masks"].astype(np.float32)
            scores = z["scores"].astype(np.float32)

            order = np.argsort(-scores)[: min(args.max_masks, len(scores))]

            score_maps = np.full((7, H, W), -1e9, dtype=np.float32)
            score_maps[0, :, :] = 0.0  # background baseline（保持 v2 原逻辑）

            for i in order:
                m = masks[i]
                if int((m > 0.5).sum()) < args.min_area:
                    continue

                box = mask_to_box(m)
                if box is None:
                    continue

                x0, y0, x1, y1 = pad_box(box, H, W, args.pad)
                crop = img_rgb[y0:y1, x0:x1]

                inp = preprocess(Image.fromarray(crop, mode="RGB")).unsqueeze(0).to(device)
                img_feat = model.encode_image(inp)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                sims = (img_feat @ text_feat.T).squeeze(0).float().cpu().numpy()  # [7]
                if args.force_pos:
                    sims = sims - float(sims.min())

                w = float(scores[i]) ** float(args.score_gamma)

                mm = (m > 0.5)
                ys, xs = np.where(mm)
                if ys.size == 0:
                    continue

                for c in range(7):
                    val = w * float(sims[c])
                    if c == 6:
                        val *= float(args.agri_boost)  # ✅ only once
                    score_maps[c, ys, xs] = np.maximum(score_maps[c, ys, xs], val)

            pred = score_maps.argmax(axis=0).astype(np.uint8) + 1
            cm += fast_confusion_matrix(gt, pred, num_classes=7, ignore_label=0)

            if args.save_pred and (idx - args.start) < 50:
                Image.fromarray(pred).save(pred_dir / f"{idx:05d}.png")

            if (j + 1) % 20 == 0:
                miou, ious = miou_from_cm(cm)
                print(f"[{idx}] mIoU={miou:.4f} per-class={np.round(ious, 4)}")

    miou, ious = miou_from_cm(cm)
    print("\n=== FINAL (MaskCLIP v2) ===")
    print("mIoU:", miou)
    print("IoU per class:", ious)

    labels = np.array(["background", "building", "road", "water", "barren", "forest", "agriculture"], dtype=object)
    np.savez_compressed(out_dir / "confmat_maskclip_v2.npz", confmat=cm, labels=labels)
    print("Saved:", str(out_dir / "confmat_maskclip_v2.npz"))


if __name__ == "__main__":
    main()
