import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image, ImageDraw
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# LoveDA label: 0 ignore, 1..7 classes
PALETTE = {
    0: (0, 0, 0),          # ignore
    1: (128, 128, 128),    # background
    2: (220, 20, 60),      # building
    3: (255, 140, 0),      # road
    4: (30, 144, 255),     # water
    5: (210, 180, 140),    # barren
    6: (34, 139, 34),      # forest
    7: (255, 215, 0),      # agriculture
}

LOVEDA_CLASSES = [
    ("background",  ["background", "land surface background"]),
    ("building",    ["building", "house", "roof"]),
    ("road",        ["road", "street", "paved road"]),
    ("water",       ["water", "river", "lake", "water body"]),
    ("barren",      ["barren land", "bare soil", "sand", "rock"]),
    ("forest",      ["forest", "trees", "woodland"]),
    ("agriculture", ["farmland", "cropland", "agriculture field"]),
]


def colorize_mask(mask: np.ndarray) -> Image.Image:
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in PALETTE.items():
        rgb[mask == k] = np.array(color, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def add_title(img: Image.Image, title: str) -> Image.Image:
    pad = 28
    w, h = img.size
    canvas = Image.new("RGB", (w, h + pad), (255, 255, 255))
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(0, 0, 0))
    return canvas


def read_lmdb_item(lmdb_path: str, idx: int):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        raw = txn.get(str(idx).encode())
    if raw is None:
        raise KeyError(f"LMDB item {idx} not found")

    obj = pickle.loads(raw)
    img_bytes, img_shape = obj[0], obj[1]   # img: CHW=(3,512,512)
    lbl_bytes, lbl_shape = obj[2], obj[3]   # lbl: HW=(512,512)

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape).copy()
    lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
    return img, lbl


def chw_bgr_to_pil_rgb(img_chw: np.ndarray) -> Image.Image:
    """LMDB image is CHW and stored as BGR -> swap to RGB -> convert to PIL."""
    img_chw = np.asarray(img_chw).astype(np.uint8)
    img_chw = img_chw[[2, 1, 0], :, :]          # BGR -> RGB
    img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H,W,3)
    return Image.fromarray(img_hwc, mode="RGB")


@torch.inference_mode()
def predict_one(processor: Sam3Processor, image_pil: Image.Image) -> np.ndarray:
    """
    Text-prompt per class -> score map -> argmax -> pred labels in 1..7
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

            m = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks, dtype=np.float32)
            s = scores.detach().float().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores, dtype=np.float32)

            if m.ndim == 4 and m.shape[1] == 1:
                m = m[:, 0, :, :]

            prompt_map = (m * s[:, None, None]).max(axis=0)
            best_map = prompt_map if best_map is None else np.maximum(best_map, prompt_map)

        if best_map is None:
            best_map = np.zeros((h, w), dtype=np.float32)

        score_maps[cls_idx] = best_map

    pred = score_maps.argmax(axis=0).astype(np.uint8) + 1
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", required=True, help="output png path (triptych)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_chw, gt = read_lmdb_item(args.lmdb, args.idx)
    img = chw_bgr_to_pil_rgb(img_chw)

    # build SAM3
    try:
        model = build_sam3_image_model(checkpoint=args.ckpt)
    except TypeError:
        model = build_sam3_image_model(checkpoint_path=args.ckpt)
    processor = Sam3Processor(model)

    pred = predict_one(processor, img)

    # --- sanity check prints ---
    u_gt, c_gt = np.unique(gt, return_counts=True)
    u_pr, c_pr = np.unique(pred, return_counts=True)
    print("GT unique:", list(zip(u_gt.tolist(), c_gt.tolist()))[:20])
    print("Pred unique:", list(zip(u_pr.tolist(), c_pr.tolist()))[:20])

    valid = (gt != 0)
    acc = (pred[valid] == gt[valid]).mean() if valid.any() else float("nan")
    print("Pixel-acc (gt!=0):", acc)
    # --- end sanity check ---


    gt_rgb = colorize_mask(gt)
    pred_rgb = colorize_mask(pred)

    a = add_title(img,      f"Image (idx={args.idx})")
    b = add_title(gt_rgb,   "GT (color)")
    c = add_title(pred_rgb, "Pred (color)")

    W, H = a.size
    canvas = Image.new("RGB", (W * 3, H), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (W, 0))
    canvas.paste(c, (W * 2, 0))

    canvas.save(out_path)
    print("saved:", str(out_path))


if __name__ == "__main__":
    main()
