# DOFA-CLIP（Training-free）在 LoveDA val_0_200 的评测

本目录包含一个评测脚本：基于已有的 mask proposals（npz），用 DOFA-CLIP 给每个 mask 判类别，再融合成像素级预测，计算 OA / mIoU。

## 输入（本地路径）
- LoveDA 验证集 LMDB：
  `/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb`
- mask proposals（来自之前 CLIP/SAM3 流程生成的 npz，0-199）：
  `/mnt/e/ovss_project/pythonProject1/repos/clip/runs/val_0_200/masks_npz/`

## 输出（本地路径）
- 混淆矩阵（confmat）：
  `/mnt/e/ovss_project/pythonProject1/runs/dofa_clip_loveda_val_0_200/confmat.npz`

## 运行方法
1) 激活环境（你当前使用的）：
   `conda activate dofa`

2) 运行评测：
   `python scripts/dofa_clip/eval_dofa_on_masks_0_200.py`

## 参考结果（val_0_200）
- OA = 0.6661
- mIoU = 0.4519

