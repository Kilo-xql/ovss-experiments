# debug_scripts/summarize_npz_segments.py
import glob, re, os, sys
import numpy as np

def metrics_from_cm(cm: np.ndarray):
    # 支持 7x7（无ignore）或 8x8（含ignore=0）
    if cm.shape == (7, 7):
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = np.where(tp+fp+fn > 0, tp/(tp+fp+fn+1e-12), np.nan)
        miou = float(np.nanmean(iou))
        oa = float(tp.sum()/(cm.sum()+1e-12))
        return oa, miou, iou, None
    elif cm.shape == (8, 8):
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = np.where(tp+fp+fn > 0, tp/(tp+fp+fn+1e-12), np.nan)
        miou_no0 = float(np.nanmean(iou[1:]))
        oa_all = float(tp.sum()/(cm.sum()+1e-12))
        oa_no0 = float(np.trace(cm[1:,1:])/(cm[1:,:].sum()+1e-12))
        return oa_no0, miou_no0, iou, oa_all
    else:
        raise ValueError(f"Unsupported cm shape: {cm.shape}")

def main():
    # 用法1：不给参数 -> 默认扫 /home/qile/ovss_runs 下面所有 loveda_confmat_*_*.npz
    # 用法2：给一个目录 -> 扫该目录下的 loveda_confmat_*_*.npz（不递归）
    if len(sys.argv) == 1:
        pattern = "/home/qile/ovss_runs/**/loveda_confmat_*_*.npz"
        paths = glob.glob(pattern, recursive=True)
    else:
        d = sys.argv[1]
        paths = glob.glob(os.path.join(d, "loveda_confmat_*_*.npz"))

    segs = []
    for p in paths:
        m = re.search(r"confmat_(\d+)_(\d+)\.npz$", p)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        d = np.load(p, allow_pickle=True)
        cm = d["conf_fuse"].astype(np.int64)
        labels = d["labels"].tolist() if "labels" in d else None
        oa, miou, iou, oa_all = metrics_from_cm(cm)
        gt = cm.sum(1).astype(np.int64).tolist()
        pred = cm.sum(0).astype(np.int64).tolist()
        segs.append((s, e, oa, miou, oa_all, gt, pred, labels, p))

    segs.sort(key=lambda x: (x[0], x[1]))
    print("found:", len(segs))
    for s,e,oa,miou,oa_all,gt,pred,labels,p in segs:
        tail = p[-60:] if len(p) > 60 else p
        if oa_all is None:
            print(f"{s:>4}-{e:<4}  OA={oa:.4f}  mIoU={miou:.4f}  {tail}")
        else:
            print(f"{s:>4}-{e:<4}  OA_no0={oa:.4f}  mIoU_no0={miou:.4f}  OA_all={oa_all:.4f}  {tail}")

if __name__ == "__main__":
    main()
