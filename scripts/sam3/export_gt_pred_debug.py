import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PALETTE = {
    0: (0, 0, 0),
    1: (128, 128, 128),
    2: (220, 20, 60),
    3: (255, 140, 0),
    4: (30, 144, 255),
    5: (210, 180, 140),
    6: (34, 139, 34),
    7: (255, 215, 0),
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

def colorize(mask: np.ndarray) -> Image.Image:
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, c in PALETTE.items():
        rgb[mask == k] = np.array(c, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def read_lmdb(lmdb_path: str, idx: int):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        raw = txn.get(str(idx).encode())
    if raw is None:
        raise KeyError(f"idx={idx} not found in lmdb")

    obj = pickle.loads(raw)
    img_bytes, img_shape = obj[0], obj[1]     # CHW
    lbl_bytes, lbl_shape = obj[2], obj[3]     # HW
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape).copy()
    gt  = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
    return img, gt

def chw_bgr_to_pil_rgb(img_chw: np.ndarray) -> Image.Image:
    img_chw = np.asarray(img_chw).astype(np.uint8)
    img_chw = img_chw[[2,1,0], :, :]          # BGR->RGB
    img_hwc = np.transpose(img_chw, (1,2,0))  # HWC
    return Image.fromarray(img_hwc, mode="RGB")

@torch.inference_mode()
def predict(processor: Sam3Processor, image_pil: Image.Image) -> np.ndarray:
    state = processor.set_image(image_pil)
    w, h = image_pil.size
    score_maps = np.zeros((len(LOVEDA_CLASSES), h, w), dtype=np.float32)

    for cls_idx, (_name, prompts) in enumerate(LOVEDA_CLASSES):
        best = None
        for p in prompts:
            out = processor.set_text_prompt(state=state, prompt=p)
            masks = out["masks"]
            scores = out["scores"]
            if masks is None or len(masks) == 0:
                continue

            m = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks, dtype=np.float32)
            s = scores.detach().float().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores, dtype=np.float32)

            if m.ndim == 4 and m.shape[1] == 1:
                m = m[:,0,:,:]

            prompt_map = (m * s[:,None,None]).max(axis=0)
            best = prompt_map if best is None else np.maximum(best, prompt_map)

        if best is None:
            best = np.zeros((h,w), dtype=np.float32)
        score_maps[cls_idx] = best

    pred = score_maps.argmax(axis=0).astype(np.uint8) + 1
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_chw, gt = read_lmdb(args.lmdb, args.idx)
    img_pil = chw_bgr_to_pil_rgb(img_chw)

    # build model
    try:
        model = build_sam3_image_model(checkpoint=args.ckpt)
    except TypeError:
        model = build_sam3_image_model(checkpoint_path=args.ckpt)
    processor = Sam3Processor(model)

    pred = predict(processor, img_pil)

    # 保存“证据文件”
    img_pil.save(out_dir / f"idx{args.idx:04d}_image.png")

    Image.fromarray(gt.astype(np.uint8), mode="L").save(out_dir / f"idx{args.idx:04d}_gt_label.png")
    colorize(gt).save(out_dir / f"idx{args.idx:04d}_gt_color.png")

    Image.fromarray(pred.astype(np.uint8), mode="L").save(out_dir / f"idx{args.idx:04d}_pred_label.png")
    colorize(pred).save(out_dir / f"idx{args.idx:04d}_pred_color.png")

    # 打印统计（可选）
    u_gt, c_gt = np.unique(gt, return_counts=True)
    u_pr, c_pr = np.unique(pred, return_counts=True)
    print("GT unique:", list(zip(u_gt.tolist(), c_gt.tolist()))[:20])
    print("Pred unique:", list(zip(u_pr.tolist(), c_pr.tolist()))[:20])

    valid = (gt != 0)
    acc = (pred[valid] == gt[valid]).mean() if valid.any() else float("nan")
    print("Pixel-acc (gt!=0):", acc)

    print("saved to:", str(out_dir))

if __name__ == "__main__":
    main()
