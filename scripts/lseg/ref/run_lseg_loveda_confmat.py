#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run LSeg on LoveDA LMDB patches and save confusion matrix stats (.npz).

LoveDA LMDB convention (aligned with dataset/lmdb_dataset_loveda.py):
- data_bytes -> reshape(data_shape) gives CHW uint8
- then channel reorder [[2,1,0]] (repo loader behavior)
- label bytes are uint8 in {1..7} (may include 0); eval ids: 0..6; ignore=255

Supports:
- prompt_set: v2 / v3
- agg: max / mean / lse
- optional per-class bias (add to logits7)

CRITICAL:
- This script will try to load ckpt with multiple strategies.
- By default, it requires exact match (missing=0 AND unexpected=0),
  otherwise it will STOP because evaluation would be meaningless.

Example:
  conda activate lseg
  cd /mnt/e/ovss_project/pythonProject1/repos/LandSegmenter

  python run_lseg_loveda_confmat.py \
    --lmdb_file /mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb \
    --start_idx 5000 --end_idx 6000 \
    --stats_dir /home/qile/ovss_runs/tmp \
    --ckpt /home/qile/ovss_runs/lseg_finetune_v4/lseg_loveda_head_ep1.pth \
    --prompt_set v2 --agg max --no_bias
"""

import argparse
import math
import sys
from pathlib import Path

import lmdb
import numpy as np
import pickle
import torch
import torchvision.transforms as T
from tqdm import tqdm


H = 512
W = 512

# 7 semantic classes: 0..6, ignore=255
NCLS = 7
IGNORE_LABEL = 255


class LoveDALMDBDataset:
    def __init__(self, lmdb_file: str):
        self.lmdb_file = lmdb_file
        self.env = lmdb.open(
            lmdb_file,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()["entries"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            raw = txn.get(str(idx).encode("utf-8"))
            if raw is None:
                raise IndexError(f"Index {idx} not found in LMDB.")

        obj = pickle.loads(raw)
        if not (isinstance(obj, tuple) and len(obj) == 5):
            raise RuntimeError(f"Unexpected record format at idx={idx}: type={type(obj)} len={len(obj) if hasattr(obj,'__len__') else 'NA'}")

        data_bytes, data_shape, lbl_bytes, lbl_shape, meta = obj

        img = np.frombuffer(data_bytes, dtype=np.uint8).reshape(data_shape).astype(np.float32)  # CHW
        if img.shape != (3, H, W):
            raise RuntimeError(f"Unexpected image shape at idx={idx}: {img.shape} (expected (3,{H},{W}))")
        img = img[[2, 1, 0], :, :]  # match repo loader
        x = torch.from_numpy(img / 255.0)  # CHW float [0,1]

        y_raw = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
        if y_raw.shape != (H, W):
            raise RuntimeError(f"Unexpected label shape at idx={idx}: {y_raw.shape} (expected ({H},{W}))")
        y = np.where(y_raw > 0, y_raw - 1, IGNORE_LABEL).astype(np.int64)
        y = torch.from_numpy(y)

        return x, y


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
    total = conf.sum() + 1e-12
    oa = np.trace(conf) / total

    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-12)
    miou = float(np.mean(iou))
    return float(oa), miou, iou


def lse_pool(x: torch.Tensor, dim: int, tau: float):
    """
    LogSumExp pooling (smooth max) over dim.
    Returns shape with keepdim=True at pooled dim.
    """
    tau = float(tau)
    if tau <= 0:
        # tau -> 0 approximates mean
        return x.mean(dim=dim, keepdim=True)
    # (1/tau) * (logsumexp(tau*x) - log(T))
    Tn = x.size(dim)
    return (torch.logsumexp(x * tau, dim=dim, keepdim=True) - math.log(Tn)) / tau


def parse_bias(bias_str: str):
    """
    bias can be:
      - single float: "0.0" (applied to all 7 classes)
      - 7 floats: "b0,b1,b2,b3,b4,b5,b6"
    """
    s = bias_str.strip()
    if "," not in s:
        v = float(s)
        return [v] * 7
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 7:
        raise ValueError("bias must be a single float or 7 comma-separated floats.")
    return [float(p) for p in parts]


def get_label_groups(prompt_set: str):
    if prompt_set == "v2":
        return [
            ["background region in an aerial image", "grass", "grassland", "vegetation", "green vegetation", "meadow",
             "aerial image of grass", "satellite image of grass", "remote sensing image of grassland",
             "vegetated ground", "plants on the ground", "vegetation background in a satellite image"],
            ["building in an aerial image", "building", "buildings", "house", "houses", "roof", "rooftop",
             "urban buildings in a satellite image", "aerial photo of buildings", "constructed area",
             "built-up area", "residential buildings"],
            ["road in an aerial image", "road", "roads", "street", "streets", "highway", "paved road",
             "asphalt road", "road network in a satellite image", "transportation road",
             "aerial photo of a road", "thin road lines in an aerial image"],
            ["water body in an aerial image", "water", "water body", "river", "lake", "pond", "stream",
             "reservoir", "aerial image of water", "satellite image of water",
             "blue water surface in a satellite image", "water region in an aerial image"],
            ["barren land in an aerial image", "barren land", "barren", "bare soil", "bare ground", "dry soil",
             "sand", "rock", "desert", "bare earth surface", "non-vegetated ground",
             "satellite image of barren land"],
            ["forest in an aerial image", "forest", "trees", "tree", "woodland", "dense forest", "tree canopy",
             "green forest in a satellite image", "aerial image of forest", "remote sensing image of trees",
             "forest region in an aerial image", "evergreen forest"],
            ["agriculture / farmland in an aerial image", "farmland", "cropland", "agriculture", "agricultural field",
             "crop field", "cultivated land", "farm field patterns", "satellite image of farmland",
             "aerial image of crop fields", "planted crops", "crop rows in an aerial image"],
        ]

    if prompt_set == "v3":
        return [
            ["grass", "grassland", "vegetation", "green vegetation", "meadow",
             "aerial image of grass", "satellite image of grass", "remote sensing image of grassland",
             "vegetated ground", "plants on the ground",
             "background region in an aerial image", "vegetation background in a satellite image"],
            ["building", "buildings", "house", "houses", "roof", "rooftop",
             "urban buildings in a satellite image", "aerial photo of buildings", "constructed area",
             "built-up area", "residential buildings", "industrial buildings"],
            ["road", "roads", "street", "streets", "highway", "paved road", "asphalt road",
             "road network in a satellite image", "transportation road", "aerial photo of a road",
             "thin road lines in an aerial image", "vehicle road surface"],
            ["water", "water body", "river", "lake", "pond", "stream", "reservoir",
             "aerial image of water", "satellite image of water", "blue water surface in a satellite image",
             "water region in an aerial image", "flowing river in a remote sensing image"],
            ["barren land", "barren", "bare soil", "bare ground", "dry soil", "sand", "rock", "desert",
             "bare earth surface", "non-vegetated ground", "aerial image of bare soil",
             "satellite image of barren land"],
            ["forest", "trees", "tree", "woodland", "dense forest", "tree canopy",
             "green forest in a satellite image", "aerial image of forest", "remote sensing image of trees",
             "forest region in an aerial image", "tall trees", "evergreen forest"],
            ["farmland", "cropland", "agriculture", "agricultural field", "crop field", "cultivated land",
             "farm field patterns", "satellite image of farmland", "aerial image of crop fields",
             "planted crops", "harvested field", "crop rows in an aerial image"],
        ]

    raise ValueError("prompt_set must be v2 or v3")


def _extract_state_dict(obj):
    """
    Return a list of plausible state_dict candidates (dict[str, Tensor]) from a loaded ckpt object.
    """
    cands = []
    if isinstance(obj, dict):
        # lightning-style
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            cands.append(obj["state_dict"])
        # common training scripts
        for k in ["model", "net", "module", "params"]:
            if k in obj and isinstance(obj[k], dict):
                cands.append(obj[k])
        # maybe already a state_dict
        # (only accept if looks like param dict)
        if all(isinstance(k, str) for k in obj.keys()):
            has_tensor = any(torch.is_tensor(v) for v in obj.values())
            if has_tensor:
                cands.append(obj)
    return cands


def _strip_prefixes(sd: dict, prefixes: tuple):
    if not prefixes:
        return sd
    out = {}
    for k, v in sd.items():
        kk = k
        changed = True
        # strip repeatedly (some keys have "module.module.")
        while changed:
            changed = False
            for pre in prefixes:
                if kk.startswith(pre):
                    kk = kk[len(pre):]
                    changed = True
        out[kk] = v
    return out


def smart_load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, allow_mismatch: bool = False):
    """
    Try multiple extraction + prefix-stripping strategies.
    Pick the best (minimum missing+unexpected).
    Default requires exact match (0,0), otherwise STOP unless allow_mismatch=True.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cands = _extract_state_dict(obj)
    if not cands:
        raise RuntimeError(f"Cannot extract state_dict from ckpt: {ckpt_path}")

    prefix_trials = [
        tuple(),  # no strip
        ("module.",),
        ("model.",),
        ("net.",),
        ("pretrained.",),
        ("pretrained.model.",),
        ("module.", "model.", "net."),
        ("net.", "model.", "module.", "pretrained.", "pretrained.model."),
    ]

    best = None
    best_info = None

    for base_sd in cands:
        for pref in prefix_trials:
            sd = _strip_prefixes(base_sd, pref)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            score = len(missing) + len(unexpected)
            info = (len(missing), len(unexpected), pref, missing[:5], unexpected[:5])
            # restore model weights? load_state_dict already modified model; to avoid accumulating,
            # we reload original weights later is hard. BUT: score comparison only; final load we redo at end.
            # So we must NOT keep these partial loads. We'll just record best and redo final load from scratch outside.
            # Workaround: use a fresh model copy? Too heavy.
            # Practical: we re-load the best again at end (this is fine even if intermediate loads happened).
            if best is None or score < best:
                best = score
                best_info = (base_sd, info)

    base_sd, info = best_info
    m_cnt, u_cnt, pref, m_ex, u_ex = info
    # final apply best
    sd_final = _strip_prefixes(base_sd, pref)
    missing, unexpected = model.load_state_dict(sd_final, strict=False)

    print("LOAD:", ckpt_path)
    print("  best_prefix_strip:", pref if pref else "(none)")
    print("  missing:", len(missing), "unexpected:", len(unexpected))
    if len(missing) or len(unexpected):
        print("  Example missing:", list(missing)[:8])
        print("  Example unexpected:", list(unexpected)[:8])
        if not allow_mismatch:
            raise SystemExit(
                "Stop: ckpt keys do not match model (missing/unexpected not zero). "
                "Evaluation would be meaningless. Use --allow_mismatch only if you know what you're doing."
            )
    else:
        print("  LOAD OK (exact match).")

    return len(missing), len(unexpected), pref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb_file", required=True)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    ap.add_argument("--stats_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lseg_repo", default="/mnt/e/ovss_project/pythonProject1/repos/lang-seg")
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--prompt_set", choices=["v2", "v3"], default="v2")
    ap.add_argument("--agg", choices=["max", "mean", "lse"], default="max")
    ap.add_argument("--lse_tau", type=float, default=10.0)

    ap.add_argument("--bias", type=str, default="0.0", help="single float or 7 floats: b0,b1,...,b6 (add to logits)")
    ap.add_argument("--no_bias", action="store_true")

    # backward-compat
    ap.add_argument("--no_bg_bias", action="store_true", help=argparse.SUPPRESS)

    # IMPORTANT: default is strict (no mismatch)
    ap.add_argument("--allow_mismatch", action="store_true", help="Allow missing/unexpected keys when loading ckpt (NOT recommended).")

    args = ap.parse_args()

    if args.no_bg_bias:
        args.no_bias = True

    # Import LSeg
    lseg_repo = Path(args.lseg_repo).resolve()
    sys.path.insert(0, str(lseg_repo))
    from modules.models.lseg_net import LSegNet  # noqa

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # prompts
    label_groups = get_label_groups(args.prompt_set)
    flat_prompts = [p for g in label_groups for p in g]
    group_sizes = [len(g) for g in label_groups]
    print(f"prompts: total={len(flat_prompts)}  group_sizes={group_sizes}  agg={args.agg}  prompt_set={args.prompt_set}")

    # model
    model = LSegNet(
        labels=flat_prompts,
        backbone="clip_vitl16_384",
        features=256,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
        crop_size=512,
    ).to(device)

    miss_n, unexp_n, used_pref = smart_load_ckpt_into_model(model, args.ckpt, allow_mismatch=args.allow_mismatch)
    model.eval()

    dataset = LoveDALMDBDataset(args.lmdb_file)
    n = len(dataset)
    start = max(0, args.start_idx)
    end = n if args.end_idx < 0 else min(n, args.end_idx)

    conf = np.zeros((NCLS, NCLS), dtype=np.int64)

    norm = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    # slices per class
    slices = []
    s = 0
    for k in group_sizes:
        slices.append((s, s + k))
        s += k

    bias_vec = parse_bias(args.bias)
    bias_on = (not args.no_bias) and any(abs(b) > 1e-12 for b in bias_vec)
    print("bias_on:", bias_on, "bias_vec:", bias_vec)

    with torch.no_grad():
        for i in tqdm(range(start, end), desc=f"LSeg LoveDA [{start}:{end}]"):
            x, y = dataset[i]
            x = norm(x).unsqueeze(0).to(device, non_blocking=True)

            out = model(x)  # (1, M, H, W)

            logits7 = []
            for (a, b) in slices:
                block = out[:, a:b, :, :]
                if args.agg == "max":
                    agg = block.max(dim=1, keepdim=True).values
                elif args.agg == "mean":
                    agg = block.mean(dim=1, keepdim=True)
                else:
                    agg = lse_pool(block, dim=1, tau=float(args.lse_tau))
                logits7.append(agg)

            out7 = torch.cat(logits7, dim=1)  # (1,7,H,W)

            if bias_on:
                bv = torch.tensor(bias_vec, device=out7.device, dtype=out7.dtype).view(1, 7, 1, 1)
                out7 = out7 + bv

            pred = out7.argmax(dim=1)[0].cpu()
            confmat_update(conf, y, pred, NCLS, ignore_label=IGNORE_LABEL)

    oa, miou, iou = metrics_from_conf(conf)
    print(f"LSeg metrics: mIoU={miou:.4f} OA={oa:.4f} (ignore=255 excluded; classes=0..6)")
    print("IoU per class [bg, bld, road, water, barren, forest, agri]:", np.round(iou, 4).tolist())

    stats_dir = Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = stats_dir / f"loveda_confmat_{start}_{end}.npz"

    np.savez_compressed(
        out_path,
        conf_clip=conf,
        conf_weak=conf,
        conf_fuse=conf,
        labels=np.array(["background", "building", "road", "water", "barren", "forest", "agriculture"], dtype=object),
        lseg_text_labels=np.array(label_groups, dtype=object),
        lseg_flat_prompts=np.array(flat_prompts, dtype=object),
        ignore_label=np.array(IGNORE_LABEL),
        crop_size=np.array(512),
        agg=np.array(args.agg),
        prompt_set=np.array(args.prompt_set),
        bias=np.array(bias_vec, dtype=np.float32),
        lse_tau=np.array(float(args.lse_tau)),
        ckpt=np.array(str(args.ckpt), dtype=object),
        load_missing=np.array(int(miss_n)),
        load_unexpected=np.array(int(unexp_n)),
        load_prefix=np.array(str(used_pref), dtype=object),
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
