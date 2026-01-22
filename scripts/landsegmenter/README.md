# LandSegmenter – LoveDA (LMDB) Reproduction

Goal: reproduce LandSegmenter evaluation on LoveDA LMDB and verify the pipeline by running a small segment [0, 200).

## Local paths (this machine)
- LandSegmenter repo: /mnt/e/ovss_project/pythonProject1/repos/LandSegmenter
- Data root (dataset/datasets_settings.py): /mnt/e/ovss_project/pythonProject1/data
- Expected LoveDA LMDB: /mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb
- Output dir: /mnt/e/ovss_project/pythonProject1/ovss_runs/repro_landseg_0_200

## What is in this folder
- patches/landsegmenter_repro_only.patch: minimal diffs for segmented eval and saving confusion matrices (.npz)
- ref/: snapshots of key modified files
- run/run_loveda_0_200.sh: one-click runner

## Run (0–200)
1) conda activate sam2
2) bash scripts/landsegmenter/run/run_loveda_0_200.sh

## Success criteria
- Console prints mIoU/OA
- Output folder contains loveda_confmat_0_200.npz
