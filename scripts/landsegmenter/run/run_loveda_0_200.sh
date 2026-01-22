#!/usr/bin/env bash
set -euo pipefail

# Your LandSegmenter code repo (local path)
LANDSEG_REPO="/mnt/e/ovss_project/pythonProject1/repos/LandSegmenter"

# Output dir (keep isolated from other models)
OUT_DIR="/mnt/e/ovss_project/pythonProject1/ovss_runs/repro_landseg_0_200"

mkdir -p "$OUT_DIR"
cd "$LANDSEG_REPO"

# segmented eval [0,200)
bash test_OV_lowmem.sh 0 loveda \
  --start_idx 0 --end_idx 200 \
  --stats_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run.log"

echo "Done. Outputs in: $OUT_DIR"
ls -lh "$OUT_DIR" | tail
