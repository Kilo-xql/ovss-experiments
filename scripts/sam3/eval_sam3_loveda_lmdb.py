import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# LoveDA: 7 类，GT=0 作为 ignore
# 预测输出我们用 1..7（和你之前的 LandSegmenter/LSeg 对齐）
LOVEDA_CLASSES = [
    ("background",  ["background", "land surface background"]),
    ("building",    ["building", "house", "roof"]),
    ("road",        ["road", "street", "paved road"]),
    ("water",       ["water", "river", "lake", "water body"]),
    ("barren",      ["barren land", "bare soil", "sand", "rock"]),
    ("forest",      ["forest", "trees", "woodland"]),
    ("agriculture", ["farmland", "cropland", "agriculture field"]),
]


def fast_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_label: int = 0):
    """cm: rows=true, cols=pred; labels in [1..num_classes], ignore_label ignored"""
    mask = (y_true != ignore_label)
    yt = y_true[mask].astype(np.int64)
    yp = y_pred[mask].astype(np.int64)

    valid = (yt >= 1) & (yt <= num_classes) & (yp >= 1) & (yp <= num_classes)
    yt = yt[valid] - 1
    yp = yp[valid] - 1

    k = num_classes
    cm = np.bincount(k * yt + yp, minlength=k * k).reshape(k, k)
    return cm


def miou_from_cm(cm: np.ndarray):
    diag = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag
    iou = np.divide(diag, union, out=np.zeros_like(diag), where=(union > 0))
    miou = float(np.mean(iou)) if len(iou) else 0.0
    return miou, iou

def _to_rgb_pil(img_np: np.ndarray) -> Image.Image:
    """LMDB里是 CHW=(3,512,512)，先做 BGR->RGB 再转 PIL."""
    img_np = np.asarray(img_np).astype(np.uint8)

    # CHW -> (先换通道) -> HWC
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        # 关键：把 (B,G,R) 交换成 (R,G,B)
        img_np = img_np[[2, 1, 0], :, :]
        img_np = np.transpose(img_np, (1, 2, 0))  # (H,W,3)
        return Image.fromarray(img_np, mode="RGB")

    # 兜底：如果未来换成 HWC，那就对最后一维 swap
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        img_np = img_np[:, :, [2, 1, 0]]
        return Image.fromarray(img_np, mode="RGB")

    # 灰度
    if img_np.ndim == 2:
        return Image.fromarray(img_np, mode="L").convert("RGB")

    raise ValueError(f"Unsupported image shape: {img_np.shape}")




@torch.inference_mode()
def predict_semantic(processor: Sam3Processor, image_pil: Image.Image):
    """
    对 7 类分别 text prompt，融合成 score_maps，然后 argmax -> 1..7
    """
    state = processor.set_image(image_pil)
    w, h = image_pil.size
    score_maps = np.zeros((len(LOVEDA_CLASSES), h, w), dtype=np.float32)

    for cls_idx, (_name, prompts) in enumerate(LOVEDA_CLASSES):
        best_map = None

        for p in prompts:
            out = processor.set_text_prompt(state=state, prompt=p)
            masks = out["masks"]
            scores = out["scores"]

            if masks is None or len(masks) == 0:
                continue

            if torch.is_tensor(masks):
                m = masks.detach().float().cpu().numpy()
            else:
                m = np.asarray(masks, dtype=np.float32)

            if torch.is_tensor(scores):
                s = scores.detach().float().cpu().numpy()
            else:
                s = np.asarray(scores, dtype=np.float32)

            # 统一 m 形状到 (N,H,W)
            if m.ndim == 4 and m.shape[1] == 1:
                m = m[:, 0, :, :]

            # 每个实例 mask * score，然后实例取 max
            prompt_map = (m * s[:, None, None]).max(axis=0)

            best_map = prompt_map if best_map is None else np.maximum(best_map, prompt_map)

        if best_map is None:
            best_map = np.zeros((h, w), dtype=np.float32)

        score_maps[cls_idx] = best_map

    pred = score_maps.argmax(axis=0).astype(np.uint8) + 1
    return pred


def read_lmdb_item(txn, idx: int):
    """
    读取你工程里常见的 LMDB 数据格式：
      obj[0]=img_bytes, obj[1]=img_shape, obj[2]=lbl_bytes, obj[3]=lbl_shape
    """
    raw = txn.get(str(idx).encode())
    if raw is None:
        return None

    obj = pickle.loads(raw)
    img_bytes, img_shape = obj[0], obj[1]
    lbl_bytes, lbl_shape = obj[2], obj[3]

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape).copy()
    lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
    return img, lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="LoveDA val LMDB path")
    ap.add_argument("--ckpt", required=True, help="sam3.pt path")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=6505)
    ap.add_argument("--save_pred", action="store_true", help="save some pred pngs for debugging")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pred:
        (out_dir / "pred_png").mkdir(exist_ok=True)

    # build model (加载本地 sam3.pt)
    try:
        model = build_sam3_image_model(checkpoint=args.ckpt)
    except TypeError:
        model = build_sam3_image_model(checkpoint_path=args.ckpt)

    processor = Sam3Processor(model)

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    cm = np.zeros((7, 7), dtype=np.int64)

    with env.begin() as txn:
        for j, idx in enumerate(range(args.start, args.end)):
            item = read_lmdb_item(txn, idx)
            if item is None:
                print(f"[WARN] missing idx={idx}, skip")
                continue

            img_np, gt = item
            image = _to_rgb_pil(img_np)
            pred = predict_semantic(processor, image)

            cm += fast_confusion_matrix(gt, pred, num_classes=7, ignore_label=0)

            # 保存少量预测用于快速检查（前 50 张）
            if args.save_pred and (idx - args.start) < 50:
                Image.fromarray(pred).save(out_dir / "pred_png" / f"{idx:05d}.png")

            if (j + 1) % 20 == 0:
                miou, ious = miou_from_cm(cm)
                print(f"[{idx}] mIoU={miou:.4f} per-class={np.round(ious, 4)}")

    miou, ious = miou_from_cm(cm)
    print("\n=== FINAL ===")
    print("mIoU:", miou)
    print("IoU per class (background..agriculture):", ious)

    labels = [x[0] for x in LOVEDA_CLASSES]
    np.savez_compressed(out_dir / "confmat.npz", confmat=cm, labels=np.array(labels, dtype=object))
    print("Saved:", str(out_dir / "confmat.npz"))


if __name__ == "__main__":
    main()
