# SegEarth-OV-3（Training-free）— LoveDA（LMDB）评估记录

本目录记录了在 **不训练（training-free）** 的前提下，使用 SegEarth-OV-3 在 LoveDA（LMDB 格式）上的评估脚本与关键配置，
并包含一个简单的 **prompt 改写**（background 增强为 background/terrain/ground/land/open area），用于提升 “background” 类的表现。

## 文件说明
- `eval_segearthov3_loveda_lmdb.py`：LoveDA LMDB 评估脚本（读 LMDB、推理、算 OA/mIoU、可保存预测 PNG）
- `cls_loveda_bgterrain.txt`：用于评估的类别 prompt（已增强 background）
- `metrics_val_0_200_bgterrain.txt`：样本区间 [0,200) 的评估输出（便于对照与复现）

## 环境
运行环境：conda `segearthov3_fix`（PyTorch + CUDA 11.8）

## 数据
LMDB 示例路径（按你当前机器路径）：
`/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb`

## 运行方法
在 **SegEarth-OV-3 仓库根目录**执行：

```bash
conda activate segearthov3_fix
cd /mnt/e/ovss_project/pythonProject1/repos/SegEarth-OV-3

# 使用本目录的 bgterrain prompt（会覆盖 configs/cls_loveda.txt）
cp -v /mnt/e/ovss_project/pythonProject1/ovss-experiments/scripts/segearthov3/cls_loveda_bgterrain.txt configs/cls_loveda.txt

python eval_segearthov3_loveda_lmdb.py \
  --lmdb /mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb \
  --out_dir /mnt/e/ovss_project/pythonProject1/runs/segearthov3_loveda/val_0_200_bgterrain \
  --start 0 --end 200 \
  --save_pred

