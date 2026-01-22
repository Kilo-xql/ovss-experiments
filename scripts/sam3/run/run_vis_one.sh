#!/usr/bin/env bash
set -euo pipefail

LMDB="/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb"
CKPT="/mnt/e/ovss_project/pythonProject1/repos/SAM3/sam3.pt"
IDX="${1:-0}"
OUT="/mnt/e/ovss_project/pythonProject1/runs/sam3_loveda/vis/idx$(printf "%04d" "$IDX")_triptych.png"

mkdir -p "$(dirname "$OUT")"
conda activate sam3

python /mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/sam3/ref/vis_sam3_loveda_one.py \
  --lmdb "$LMDB" --ckpt "$CKPT" --idx "$IDX" --out "$OUT"

echo "Saved: $OUT"
