#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetune LSeg on LoveDA LMDB with:
- strict freezing (only scratch/head/decoder; backbone frozen)
- BatchNorm frozen (critical for batch=1 stability)
- bucket sampling (domain-balanced)
- optional aug + optional OHEM
- constrained checkpoint selection (do NOT hurt strong slices)

This version fixes the instability you observed: BN drift causes strong/weak to jump.
"""

import sys
import time
import argparse
from pathlib import Path

import lmdb
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Sampler


H = 512
W = 512
NCLS = 7
IGNORE_LABEL = 255


# -------------------------
# LoveDA LMDB loader (train/eval)
# -------------------------
class LoveDALMDB(Dataset):
    """
    LMDB sample format: (img_bytes, img_shape, lbl_bytes, lbl_shape, meta)
    img_shape usually (3,512,512) CHW, stored in BGR if written by cv2.
    We swap to RGB to match CLIP norm.
    """
    def __init__(self, lmdb_file: str):
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
                raise IndexError(idx)
        data_bytes, data_shape, lbl_bytes, lbl_shape, meta = pickle.loads(raw)

        img = np.frombuffer(data_bytes, dtype=np.uint8).reshape(data_shape).astype(np.float32)
        # BGR->RGB
        img = img[[2, 1, 0], :, :]
        x = torch.from_numpy(img / 255.0).float()  # (3,H,W) in [0,1]

        y_raw = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
        # GT 1..7 -> 0..6, 0 -> IGNORE(255)
        y = np.where(y_raw > 0, y_raw - 1, IGNORE_LABEL).astype(np.int64)
        y = torch.from_numpy(y).long()
        return x, y


# -------------------------
# Confusion matrix utils
# -------------------------
@torch.no_grad()
def confmat_update(conf: np.ndarray, gt: torch.Tensor, pred: torch.Tensor):
    gt = gt.view(-1)
    pred = pred.view(-1)
    valid = (gt != IGNORE_LABEL) & (gt >= 0) & (gt < NCLS) & (pred >= 0) & (pred < NCLS)
    gt = gt[valid]
    pred = pred[valid]
    idx = gt * NCLS + pred
    binc = torch.bincount(idx, minlength=NCLS * NCLS).cpu().numpy().reshape(NCLS, NCLS)
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


# -------------------------
# Prompt set (v2) : 7 classes, each 12 prompts => 84
# -------------------------
def get_label_groups_v2():
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


# -------------------------
# Bucket sampler (domain-balanced)
# -------------------------
class BucketSampler(Sampler):
    def __init__(self, length, buckets, probs, num_samples=None, seed=123):
        self.length = length
        self.buckets = []
        for (s, e) in buckets:
            s = max(0, int(s))
            e = min(int(e), length)
            if e > s:
                self.buckets.append((s, e))
        if not self.buckets:
            self.buckets = [(0, length)]
        probs = np.array(probs, dtype=np.float64)
        if probs.size != len(self.buckets):
            probs = np.ones(len(self.buckets), dtype=np.float64)
        probs = probs / probs.sum()
        self.probs = probs
        self.num_samples = int(num_samples) if num_samples is not None else length
        self.base_seed = int(seed)
        self.epoch = 0

    def set_epoch(self, ep: int):
        self.epoch = int(ep)

    def __iter__(self):
        rng = np.random.RandomState(self.base_seed + self.epoch * 10007)
        for _ in range(self.num_samples):
            b = int(rng.choice(len(self.buckets), p=self.probs))
            s, e = self.buckets[b]
            idx = int(rng.randint(s, e))
            yield idx

    def __len__(self):
        return self.num_samples


# -------------------------
# Augmentations (tensor-based)
# -------------------------
def aug_tensor(x, enable=True,
               jitter=0.20, hue=0.04,
               gray_p=0.05, blur_p=0.05,
               noise_std=0.01):
    if (not enable) or x.numel() == 0:
        return x

    B = x.shape[0]
    out = x
    for i in range(B):
        xi = out[i]
        if jitter > 0:
            b = float(torch.empty(1).uniform_(1 - jitter, 1 + jitter))
            c = float(torch.empty(1).uniform_(1 - jitter, 1 + jitter))
            s = float(torch.empty(1).uniform_(1 - jitter, 1 + jitter))
            h = float(torch.empty(1).uniform_(-hue, hue)) if hue > 0 else 0.0
            xi = TF.adjust_brightness(xi, b)
            xi = TF.adjust_contrast(xi, c)
            xi = TF.adjust_saturation(xi, s)
            if hue > 0:
                xi = TF.adjust_hue(xi, h)
        if float(torch.rand(1)) < gray_p:
            xi = TF.rgb_to_grayscale(xi, num_output_channels=3)
        if float(torch.rand(1)) < blur_p:
            xi = TF.gaussian_blur(xi, kernel_size=[3, 3], sigma=[0.1, 1.2])
        if noise_std > 0:
            xi = (xi + torch.randn_like(xi) * noise_std).clamp(0.0, 1.0)
        out[i] = xi
    return out


# -------------------------
# Loss: optional OHEM
# -------------------------
def ohem_ce_loss(logits, target, ignore_index=IGNORE_LABEL, topk=0.10):
    per = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")  # (B,H,W)
    valid = (target != ignore_index)
    per = per[valid]
    if per.numel() == 0:
        return logits.sum() * 0.0
    if topk >= 1.0:
        return per.mean()
    k = max(1, int(per.numel() * float(topk)))
    topv, _ = torch.topk(per, k, largest=True, sorted=False)
    return topv.mean()


# -------------------------
# Eval (fast, with cap)
# -------------------------
@torch.no_grad()
def eval_slice(model, lmdb_file, start, end, device, agg="max", cap=200):
    ds = LoveDALMDB(lmdb_file)
    end = min(end, len(ds))
    if end <= start:
        return 0.0, 0.0, np.zeros(NCLS, dtype=np.float64)

    n = end - start
    if cap is None or cap <= 0 or cap >= n:
        indices = list(range(start, end))
    else:
        step = max(1, n // cap)
        indices = list(range(start, end, step))[:cap]

    norm = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    conf = np.zeros((NCLS, NCLS), dtype=np.int64)
    model.eval()

    for idx in indices:
        x, y = ds[idx]
        x = norm(x).unsqueeze(0).to(device, non_blocking=True)

        out = model(x)  # (1,84,H,W)
        outs = []
        for c in range(NCLS):
            block = out[:, c * 12:(c + 1) * 12, :, :]
            if agg == "max":
                outs.append(block.max(dim=1, keepdim=True).values)
            else:
                outs.append(block.mean(dim=1, keepdim=True))
        out7 = torch.cat(outs, dim=1)  # (1,7,H,W)

        pred = out7.argmax(dim=1)[0].cpu()
        confmat_update(conf, y, pred)

    oa, miou, iou = metrics_from_conf(conf)
    return miou, oa, iou


@torch.no_grad()
def eval_multi(model, lmdb_file, slices, device, agg="max", cap=200):
    mious = []
    each = []
    for (s, e) in slices:
        miou, oa, _ = eval_slice(model, lmdb_file, s, e, device, agg=agg, cap=cap)
        mious.append(miou)
        each.append(miou)
    return float(np.mean(mious)) if mious else 0.0, each


# -------------------------
# Load checkpoint with best prefix strip
# -------------------------
def best_prefix_load(model, ckpt_path: str):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    prefixes = ("", "net.", "model.", "module.", "pretrained.", "pretrained.model.")
    best = None
    best_info = None
    for pre in prefixes:
        new_sd = {}
        for k, v in sd.items():
            kk = k
            if pre != "" and kk.startswith(pre):
                kk = kk[len(pre):]
            new_sd[kk] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if len(missing) == 0 and len(unexpected) == 0:
            return pre, 0, 0
        score = len(missing) + len(unexpected)
        if best is None or score < best:
            best = score
            best_info = (pre, len(missing), len(unexpected))
    pre, m, u = best_info
    new_sd = {}
    for k, v in sd.items():
        kk = k
        if pre != "" and kk.startswith(pre):
            kk = kk[len(pre):]
        new_sd[kk] = v
    model.load_state_dict(new_sd, strict=False)
    return pre, m, u


# -------------------------
# Freeze BN (CRITICAL for batch=1)
# -------------------------
def set_bn_eval(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_lmdb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lseg_repo", default="/mnt/e/ovss_project/pythonProject1/repos/lang-seg")
    ap.add_argument("--base_ckpt", default="/mnt/e/ovss_project/pythonProject1/repos/lang-seg/checkpoints/demo_e200.ckpt")

    ap.add_argument("--resume", default="", help="resume from epX.pth (optional)")
    ap.add_argument("--start_epoch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)

    # eval
    ap.add_argument("--eval_lmdb", default="/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb")
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--eval_cap", type=int, default=200)

    ap.add_argument("--strong_slices", type=str, default="0,1000;1000,2000;2000,2500")
    ap.add_argument("--weak_slices", type=str, default="2500,3000;3000,4000;4000,5000;5000,6000;6000,6505")
    ap.add_argument("--eps_strong", type=float, default=0.01)

    # train buckets
    ap.add_argument("--train_buckets", type=str, default="0,2500;2500,999999")
    ap.add_argument("--bucket_probs", type=str, default="0.8,0.2")

    # aug + ohem (optional)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.20)
    ap.add_argument("--hue", type=float, default=0.04)
    ap.add_argument("--gray_p", type=float, default=0.05)
    ap.add_argument("--blur_p", type=float, default=0.05)
    ap.add_argument("--noise_std", type=float, default=0.01)

    ap.add_argument("--ohem", action="store_true")
    ap.add_argument("--ohem_topk", type=float, default=0.10)

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # import LSeg
    sys.path.insert(0, str(Path(args.lseg_repo).resolve()))
    from modules.models.lseg_net import LSegNet  # noqa

    label_groups = get_label_groups_v2()
    flat_prompts = [p for g in label_groups for p in g]
    print("prompts total:", len(flat_prompts), "(expect 84)")

    model = LSegNet(
        labels=flat_prompts,
        backbone="clip_vitl16_384",
        features=256,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
        crop_size=512,
    ).to(device)

    print("Loading base:", args.base_ckpt)
    pre, m, u = best_prefix_load(model, args.base_ckpt)
    print(f"base load: prefix='{pre}' missing={m} unexpected={u}")

    if args.resume:
        print("Resuming from:", args.resume)
        pre, m, u = best_prefix_load(model, args.resume)
        print(f"resume load: prefix='{pre}' missing={m} unexpected={u}")

    # --------- STRICT FREEZE POLICY ----------
    for n, p in model.named_parameters():
        p.requires_grad = False

    # allow logit_scale
    for n, p in model.named_parameters():
        if n in ("logit_scale", "clip_pretrained.logit_scale"):
            p.requires_grad = True

    UNFREEZE_KEYS = ("scratch", "head", "decode", "decoder", "fusion", "upsample", "seg", "out_conv", "classifier")

    for n, p in model.named_parameters():
        # never unfreeze backbones
        if n.startswith("clip_pretrained.") or n.startswith("pretrained.model."):
            continue
        if any(k in n.lower() for k in UNFREEZE_KEYS):
            p.requires_grad = True

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))
    print("Trainable names (first 80):")
    print("\n".join(trainable_names[:80]))
    # ----------------------------------------

    # Freeze BN (critical)
    model.apply(set_bn_eval)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ds = LoveDALMDB(args.train_lmdb)

    def parse_pairs_list(s, sep=";"):
        out = []
        for part in s.split(sep):
            part = part.strip()
            if not part:
                continue
            a, b = part.split(",")
            out.append((int(a), int(b)))
        return out

    buckets = parse_pairs_list(args.train_buckets)
    probs = [float(x) for x in args.bucket_probs.split(",") if x.strip() != ""]
    sampler = BucketSampler(len(ds), buckets=buckets, probs=probs, num_samples=len(ds), seed=123)

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    norm = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    strong_slices = parse_pairs_list(args.strong_slices)
    weak_slices = parse_pairs_list(args.weak_slices)

    # baseline
    model.eval()
    strong_base, strong_each = eval_multi(model, args.eval_lmdb, strong_slices, device, cap=args.eval_cap)
    weak_base, weak_each = eval_multi(model, args.eval_lmdb, weak_slices, device, cap=args.eval_cap)
    print("\n[BASELINE before finetune]")
    print(" strong_slices:", strong_slices, "avg=", f"{strong_base:.4f}", "each=", [round(x, 4) for x in strong_each])
    print(" weak_slices  :", weak_slices,   "avg=", f"{weak_base:.4f}",   "each=", [round(x, 4) for x in weak_each])
    print(" constraint: strong >= baseline - eps =", f"{strong_base - args.eps_strong:.4f}")

    best_pareto = -1e9
    best_weak = -1.0
    best_avg3 = -1.0

    global_step = 0

    for ep in range(args.start_epoch, args.start_epoch + args.epochs):
        sampler.set_epoch(ep)
        t0 = time.time()

        model.train()
        model.apply(set_bn_eval)  # IMPORTANT: keep BN frozen even after train()

        opt.zero_grad(set_to_none=True)

        for it, (x, y) in enumerate(dl, start=1):
            global_step += 1

            if args.aug:
                x = aug_tensor(
                    x, enable=True,
                    jitter=args.jitter, hue=args.hue,
                    gray_p=args.gray_p, blur_p=args.blur_p,
                    noise_std=args.noise_std
                )

            x = norm(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)  # (B,84,H,W)

                outs = []
                for c in range(NCLS):
                    block = out[:, c * 12:(c + 1) * 12, :, :]
                    outs.append(block.max(dim=1, keepdim=True).values)
                out7 = torch.cat(outs, dim=1)  # (B,7,H,W)

                if args.ohem:
                    loss = ohem_ce_loss(out7, y, topk=args.ohem_topk)
                else:
                    loss = F.cross_entropy(out7, y, ignore_index=IGNORE_LABEL)

            scaler.scale(loss / args.accum).backward()

            if global_step % args.accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                print(f"epoch {ep} step {global_step} loss={float(loss):.4f}")

            if global_step % args.eval_every == 0:
                model.eval()
                strong_avg, _ = eval_multi(model, args.eval_lmdb, strong_slices, device, cap=args.eval_cap)
                weak_avg, _ = eval_multi(model, args.eval_lmdb, weak_slices, device, cap=args.eval_cap)

                # classic 3 slices
                miou_h, _, _ = eval_slice(model, args.eval_lmdb, 0, 200, device, cap=min(200, args.eval_cap))
                miou_m, _, _ = eval_slice(model, args.eval_lmdb, 4000, 4500, device, cap=min(200, args.eval_cap))
                miou_t, _, _ = eval_slice(model, args.eval_lmdb, 5000, 5200, device, cap=min(200, args.eval_cap))
                avg3 = (miou_h + miou_m + miou_t) / 3.0

                ok_strong = (strong_avg >= strong_base - args.eps_strong)

                print(
                    f"[EVAL step {global_step}] strong={strong_avg:.4f} weak={weak_avg:.4f} "
                    f"(strong_base={strong_base:.4f}, eps={args.eps_strong}) "
                    f"avg3={avg3:.4f} head={miou_h:.4f} mid={miou_m:.4f} tail={miou_t:.4f}"
                )

                # Only save checkpoints when strong constraint holds
                if ok_strong:
                    score = weak_avg * 10.0 + avg3
                    if score > best_pareto:
                        best_pareto = score
                        p = out_dir / "best_pareto.pth"
                        torch.save(model.state_dict(), p)
                        print("  saved best_pareto:", p)

                    if weak_avg > best_weak:
                        best_weak = weak_avg
                        p = out_dir / "best_weak.pth"
                        torch.save(model.state_dict(), p)
                        print("  saved best_weak:", p)

                    if avg3 > best_avg3:
                        best_avg3 = avg3
                        p = out_dir / "best_avg3.pth"
                        torch.save(model.state_dict(), p)
                        print("  saved best_avg3:", p)

                model.train()
                model.apply(set_bn_eval)

        p_last = out_dir / f"ep{ep}_last.pth"
        torch.save(model.state_dict(), p_last)
        print(f"[epoch {ep}] saved {p_last}  time={time.time()-t0:.1f}s")

    print("Done.")
    print("strong_base=", strong_base, "weak_base=", weak_base)
    print("best_pareto=", best_pareto, "best_weak=", best_weak, "best_avg3=", best_avg3)


if __name__ == "__main__":
    main()
