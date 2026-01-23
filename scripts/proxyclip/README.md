mkdir -p scripts/proxyclip

cat > scripts/proxyclip/README.md <<'MD'
# ProxyCLIP (training-free) — LoveDA (subset 0–200)

本目录用于复现我在 LoveDA 子集（0–200）上跑通的 ProxyCLIP 评测流程。  
**注意：数据与权重不上传到 Git。**

## Environment
- conda env: `proxyclip`
- versions (example):
  - torch 2.1.2+cu118, mmcv 2.1.0, mmengine 0.8.4, mmseg 1.2.2

## Dataset (NOT included)
本机路径（你的实际路径）：
`/mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/datasets_ref/loveda_lmdb_0_200`

目录结构：
- `images/`：输入影像（png）
- `labels/`：像素级标签图（png，单通道，每个像素是类别 id）
- `split.txt`：评测列表（如 00000, 00001, ...）

## Checkpoints (NOT included)
- SAM ViT-B checkpoint (example):
`/mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/checkpoints/sam/sam_vit_b_01ec64.pth`

## How to run (eval)
```bash
conda activate proxyclip
cd /mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/repo/ProxyCLIP

python eval.py \
  --config /mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/repo/ProxyCLIP/configs/cfg_loveda_lmdb_0_200.py \
  --work-dir /mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/outputs/proxyclip_loveda_0_200
