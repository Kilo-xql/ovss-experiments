# debug_scripts/gt_hist_range.py
import sys
import lmdb, pickle
import numpy as np

LMDB = "/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb"
NCLS = 8  # 0..7 (0可能是ignore/空位)，1..7是7类

def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_scripts/gt_hist_range.py <start> <end>")
        sys.exit(1)
    s = int(sys.argv[1]); e = int(sys.argv[2])

    env = lmdb.open(LMDB, readonly=True, lock=False, readahead=False, meminit=False)
    hist = np.zeros(NCLS, dtype=np.int64)

    with env.begin() as txn:
        for i in range(s, e):
            raw = txn.get(str(i).encode())
            if raw is None:
                print(f"[WARN] missing key {i}")
                continue
            obj = pickle.loads(raw)
            # loveda lmdb: (data_bytes, data_shape, lbl_bytes, lbl_shape, extra)
            lbl_bytes, lbl_shape = obj[2], obj[3]
            y = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape)
            hist += np.bincount(y.reshape(-1), minlength=NCLS)

    total = hist.sum()
    print(f"range: {s}-{e}  total_pixels={int(total)}")
    print("gt_counts:", hist.tolist())

    # 只看 1..7 的占比（如果你的GT没有0类，这里更直观）
    den = hist[1:].sum()
    if den > 0:
        frac = (hist[1:] / den).tolist()
        print("gt_frac(1..7):", [round(x, 4) for x in frac])
    else:
        print("gt_frac(1..7): all zero?")

if __name__ == "__main__":
    main()
