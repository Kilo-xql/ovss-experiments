# eval_segearthov3_loveda_lmdb.py
import os
import io
import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

import lmdb
import torch

# --- LoveDA: 7 classes (GT in LMDB is usually 0=ignore, 1..7=classes) ---
N_CLS = 7
IGNORE_LABEL = 255
LOVEDA_CLASS_NAMES = [
    "building", "road", "water", "barren", "forest", "agriculture", "background_like"
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="LoveDA val lmdb dir, contains data.mdb/lock.mdb")
    ap.add_argument("--out_dir", required=True, help="output dir for metrics/confmat/pred_png")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--save_pred", action="store_true", help="save pred label png (1..7 for viewing, 0 reserved)")
    ap.add_argument("--limit", type=int, default=None, help="optional hard cap on number of samples processed")
    ap.add_argument("--tmp_img_dir", default=None, help="tmp dir to dump lmdb images as png for model.predict(img_path)")
    ap.add_argument("--keep_tmp", action="store_true", help="do not delete tmp images after run")
    return ap.parse_args()


def _open_lmdb(lmdb_dir: str):
    # readonly + no lock => safe for parallel reads
    return lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)


def _load_item(txn, idx: int):
    raw = txn.get(str(idx).encode())
    if raw is None:
        return None
    obj = pickle.loads(raw)
    # expected: (img_bytes, img_shape, lbl_bytes, lbl_shape, ...)
    img_bytes, img_shape = obj[0], obj[1]
    lbl_bytes, lbl_shape = obj[2], obj[3]

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
    lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape)
    return img, lbl


def _img_np_to_rgb_pil(img: np.ndarray) -> Image.Image:
    """
    LMDB image in your previous pipeline was BGR.
    Accept:
      - (3,H,W) BGR CHW
      - (H,W,3) BGR HWC
    Return PIL RGB.
    """
    if img.ndim == 3 and img.shape[0] == 3:
        # CHW (BGR)
        chw = img
        rgb = chw[[2, 1, 0], :, :]  # BGR->RGB
        hwc = np.transpose(rgb, (1, 2, 0))
        return Image.fromarray(hwc, mode="RGB")

    if img.ndim == 3 and img.shape[2] == 3:
        # HWC (BGR)
        rgb = img[:, :, ::-1]  # BGR->RGB
        return Image.fromarray(rgb, mode="RGB")

    raise RuntimeError(f"Unexpected image shape: {img.shape}")


def _map_gt(lbl: np.ndarray) -> torch.Tensor:
    """
    Map GT:
      - 0 -> IGNORE_LABEL (255)
      - 1..7 -> 0..6
    Return: (H,W) int64 tensor
    """
    if lbl.ndim == 3:
        # sometimes stored as (1,H,W) or (H,W,1)
        if lbl.shape[0] == 1:
            lbl = lbl[0]
        elif lbl.shape[-1] == 1:
            lbl = lbl[..., 0]
        else:
            raise RuntimeError(f"Unexpected label shape: {lbl.shape}")

    y = np.where(lbl > 0, lbl - 1, IGNORE_LABEL).astype(np.int64)
    return torch.from_numpy(y)


def confmat_update(conf: np.ndarray, gt: torch.Tensor, pred: torch.Tensor, n_cls: int, ignore_label: int):
    gt = gt.view(-1)
    pred = pred.view(-1)
    valid = (gt != ignore_label) & (gt >= 0) & (gt < n_cls) & (pred >= 0) & (pred < n_cls)
    gt = gt[valid]
    pred = pred[valid]
    idx = gt * n_cls + pred
    binc = torch.bincount(idx, minlength=n_cls * n_cls).cpu().numpy().reshape(n_cls, n_cls)
    conf += binc


def metrics_from_conf(conf: np.ndarray):
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-12)
    miou = float(np.mean(iou))
    oa = float(tp.sum() / (conf.sum() + 1e-12))
    return oa, miou, iou


def extract_pred_label(model_out, n_cls: int) -> torch.Tensor:
    """
    Robust extractor:
      - list[SegDataSample] with pred_sem_seg.data
      - SegDataSample with pred_sem_seg.data
      - torch.Tensor logits (B,C,H,W) or (C,H,W) or label map (H,W)
    Return: (H,W) int64 label in 0..6
    """
    if isinstance(model_out, list) and len(model_out) > 0:
        model_out = model_out[0]

    if hasattr(model_out, "pred_sem_seg"):
        ps = model_out.pred_sem_seg
        if hasattr(ps, "data"):
            t = ps.data
            if torch.is_tensor(t):
                if t.ndim == 3:
                    t = t.squeeze(0)
                return t.long()

    if torch.is_tensor(model_out):
        t = model_out
        if t.ndim == 4:
            t = t.argmax(dim=1)[0]
            return t.long()
        if t.ndim == 3:
            if t.shape[0] == n_cls:
                t = t.argmax(dim=0)
            return t.long()
        if t.ndim == 2:
            return t.long()

    raise RuntimeError(f"Cannot extract pred label from type={type(model_out)}")


def build_segearth_model(device: str):
    from segearthov3_segmentor import SegEarthOV3Segmentation

    # LoveDA 类名文件在 configs/cls_loveda.txt（你已验证 build ok）
    classname_path = "./configs/cls_loveda_bgterrain.txt"
    if not Path(classname_path).exists():
        raise FileNotFoundError(f"Cannot find classname_path: {classname_path}")

    print("[INFO] Using classname_path =", str(Path(classname_path).resolve()))

    model = SegEarthOV3Segmentation(classname_path=classname_path)
    model.to(device)
    model.eval()
    return model


def _make_data_sample(img_path: str, H: int, W: int):
    """
    SegEarthOV3Segmentation.predict() 会从 data_samples[0].metainfo['img_path'] 里读图
    所以必须把 img_path 放进去。
    """
    try:
        from mmseg.structures import SegDataSample
        ds = SegDataSample()
        ds.set_metainfo(
            dict(
                img_path=img_path,
                img_shape=(H, W),
                ori_shape=(H, W),
                pad_shape=(H, W),
                scale_factor=1.0,
                flip=False,
            )
        )
        return ds
    except Exception:
        # 万一 mmseg 的结构体不可用，至少给个最小替代（有些实现只用 metainfo）
        class _Dummy:
            def __init__(self, metainfo):
                self.metainfo = metainfo

            def set_metainfo(self, x):
                self.metainfo = x

        return _Dummy(
            dict(
                img_path=img_path,
                img_shape=(H, W),
                ori_shape=(H, W),
                pad_shape=(H, W),
                scale_factor=1.0,
                flip=False,
            )
        )


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = out_dir / "pred_png"
    if args.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(args.tmp_img_dir) if args.tmp_img_dir else (out_dir / "_tmp_images")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print("[1] Build model ...")
    model = build_segearth_model(device)

    print("[2] Open LMDB ...")
    env = _open_lmdb(args.lmdb)
    conf = np.zeros((N_CLS, N_CLS), dtype=np.int64)

    processed = 0
    written_tmp = []

    with env.begin() as txn:
        for idx in range(args.start, args.end):
            item = _load_item(txn, idx)
            if item is None:
                print(f"[WARN] idx={idx} not found in LMDB, stop.")
                break

            img_np, gt_np = item

            # --- dump image to tmp png so model.predict can read it ---
            pil_img = _img_np_to_rgb_pil(img_np)
            img_path = tmp_dir / f"{idx:06d}.png"
            pil_img.save(img_path)
            written_tmp.append(img_path)

            # --- GT ---
            gt = _map_gt(gt_np)
            H, W = int(gt.shape[0]), int(gt.shape[1])

            data_sample = _make_data_sample(str(img_path), H, W)

            # --- run predict ---
            # 这个 predict 很可能签名是 predict(inputs, data_samples=...)
            # 但内部其实用 data_samples[0].metainfo['img_path'] 来读图
            try:
                out = model.predict(None, data_samples=[data_sample])
            except TypeError:
                # 兼容少数签名：predict(data_samples=...) 或 predict(img_path, data_samples=...)
                try:
                    out = model.predict(str(img_path), data_samples=[data_sample])
                except Exception as e:
                    raise RuntimeError(f"model.predict call failed: {e}")

            pred = extract_pred_label(out, N_CLS)

            confmat_update(conf, gt.cpu(), pred.cpu(), N_CLS, IGNORE_LABEL)

            if args.save_pred:
                # save as 1..7 with 0=ignore for quick viewing
                pred_vis = pred.cpu().numpy().astype(np.uint8) + 1
                Image.fromarray(pred_vis, mode="L").save(pred_dir / f"{idx:06d}.png")

            processed += 1
            if args.limit is not None and processed >= args.limit:
                break

            if processed % 10 == 0:
                oa, miou, _ = metrics_from_conf(conf)
                print(f"[{processed}] idx={idx}  OA={oa:.4f}  mIoU={miou:.4f}")

    oa, miou, iou = metrics_from_conf(conf)
    print("\n=== Done ===")
    print("processed:", processed)
    print("OA:", oa)
    print("mIoU:", miou)
    print("per-class IoU:", iou)

    np.savez(out_dir / "confmat.npz", conf=conf, oa=oa, miou=miou, iou=iou)

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"processed={processed}\n")
        f.write(f"OA={oa}\n")
        f.write(f"mIoU={miou}\n")
        f.write("IoU=" + ",".join([str(float(x)) for x in iou]) + "\n")
        f.write("classes=" + ",".join(LOVEDA_CLASS_NAMES) + "\n")

    print("saved:", str(out_dir / "confmat.npz"))
    print("saved:", str(out_dir / "metrics.txt"))

    # cleanup tmp images
    if (not args.keep_tmp) and len(written_tmp) > 0:
        for p in written_tmp:
            try:
                p.unlink()
            except Exception:
                pass
        # 目录不强删，避免误删你 out_dir 下其它东西
        print("[INFO] tmp images deleted. (use --keep_tmp to keep them)")
    else:
        print("[INFO] tmp images kept at:", str(tmp_dir))


if __name__ == "__main__":
    main()
