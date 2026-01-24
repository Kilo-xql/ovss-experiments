import os, lmdb, pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

# ===== 固定路径：不要混 =====
LMDB_PATH = "/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb"
MASK_DIR  = "/mnt/e/ovss_project/pythonProject1/repos/clip/runs/val_0_200/masks_npz"
OUT_DIR   = "/mnt/e/ovss_project/pythonProject1/runs/dofa_clip_loveda_val_0_200"
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["background","building","road","water","barren land","forest","agricultural land"]
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

def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = create_model_from_pretrained("hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO")
    tokenizer = get_tokenizer("hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO")
    model = model.to(device).eval()

    text = tokenizer(PROMPTS, context_length=model.context_length).to(device)
    with torch.no_grad():
        txt_feat = F.normalize(model.encode_text(text), dim=-1)

    wvs = torch.tensor([0.665, 0.560, 0.490], device=device)

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)

    conf_total = np.zeros((7,7), dtype=np.int64)

    for idx in range(200):
        key = str(idx).encode()
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            print("LMDB missing key:", idx)
            continue

        img_bytes, img_shape, lbl_bytes, lbl_shape = pickle.loads(raw)[:4]
        img_chw = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)   # BGR CHW
        lbl_hw  = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape)

        img_rgb = img_chw[::-1].transpose(1,2,0)  # -> RGB HWC
        gt = np.where(lbl_hw == 0, 255, lbl_hw - 1).astype(np.uint8)

        npz_path = os.path.join(MASK_DIR, f"{idx:05d}.npz")
        d = np.load(npz_path, allow_pickle=True)
        masks  = d["masks"].astype(np.float32)
        scores = d["scores"].astype(np.float32)
        order = np.argsort(-scores)

        H, W = gt.shape
        pred = np.full((H, W), 255, dtype=np.uint8)  # 未覆盖=ignore
        filled = np.zeros((H, W), dtype=bool)

        def classify_mask(mask_bool: np.ndarray) -> int:
            ys, xs = np.where(mask_bool)
            if ys.size == 0:
                return 0
            y0,y1 = ys.min(), ys.max()+1
            x0,x1 = xs.min(), xs.max()+1
            crop = img_rgb[y0:y1, x0:x1, :]
            m    = mask_bool[y0:y1, x0:x1]
            crop2 = crop.copy()
            crop2[~m] = 0
            im = preprocess(Image.fromarray(crop2)).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=="cuda")):
                img_out = model.visual.trunk(im, wvs)
                img_feat = img_out[0] if isinstance(img_out, (tuple, list)) else img_out
                img_feat = F.normalize(img_feat, dim=-1)
                sim = (img_feat @ txt_feat.T)[0]
                return int(sim.argmax().item())

        # 用全部 masks（你这批通常每张几十个）
        for i in order:
            m = masks[i] > 0.5
            cls = classify_mask(m)
            write = m & (~filled)
            pred[write] = cls
            filled[write] = True

        valid = (gt != 255) & (pred != 255)
        gtv = gt[valid].astype(np.int64)
        prv = pred[valid].astype(np.int64)

        for g, p in zip(gtv, prv):
            if 0 <= g < 7 and 0 <= p < 7:
                conf_total[g,p] += 1

        if (idx+1) % 10 == 0:
            oa, miou, _ = metrics_from_conf(conf_total)
            print(f"[{idx+1:03d}/200] OA={oa:.4f} mIoU={miou:.4f}")

    oa, miou, ious = metrics_from_conf(conf_total)
    np.savez(os.path.join(OUT_DIR, "confmat.npz"), confmat=conf_total)

    print("=== DONE 0-199 ===")
    print("OA:", oa)
    print("mIoU:", miou)
    print("IoU per class:")
    for c, v in zip(CLASSES, ious):
        print(f"  {c:18s} {v if not np.isnan(v) else 'nan'}")

    print("saved:", os.path.join(OUT_DIR, "confmat.npz"))

if __name__ == "__main__":
    main()
