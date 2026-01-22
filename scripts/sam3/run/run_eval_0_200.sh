#!/usr/bin/env bash
set -euo pipefail

# --- paths (edit here if you move folders) ---
LMDB="/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb"
CKPT="/mnt/e/ovss_project/pythonProject1/repos/SAM3/sam3.pt"
OUT_DIR="/mnt/e/ovss_project/pythonProject1/runs/sam3_loveda/small_0_200"

mkdir -p "$OUT_DIR"
conda activate sam3

python /mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/sam3/ref/eval_sam3_loveda_lmdb.py \
  --lmdb "$LMDB" --ckpt "$CKPT" --out_dir "$OUT_DIR" --start 0 --end 200

echo "Done. Check: $OUT_DIR"
ls -lh "$OUT_DIR" | tail
