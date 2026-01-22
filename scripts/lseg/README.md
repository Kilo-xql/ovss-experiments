# LSeg (LoveDA LMDB) – Workflow & Paths

目的：把 LSeg 的代码/数据/权重/输出路径梳理清楚，后续更换数据集时不会混乱。

## 关键仓库（必须分清）
- 模型实现（上游 lang-seg / LSegNet）：/mnt/e/ovss_project/pythonProject1/repos/lang-seg
- 我们的实验工作台（脚本所在）：/mnt/e/ovss_project/pythonProject1/repos/LandSegmenter

本仓库保存的是“实验脚本快照”，在 scripts/lseg/ref/。

## 数据（LMDB）
- 数据根目录：/mnt/e/ovss_project/pythonProject1/data
- LoveDA val：/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb
- LoveDA train（finetune 用）：/mnt/e/ovss_project/pythonProject1/data/ft_train/loveda_p512_train.lmdb

## 权重（不上传到 git，只记录路径）
- 默认使用（v6 stage1）：/home/qile/ovss_runs/v6_bnfreeze_stage1/best_pareto.pth

Windows 视角同一路径：
\\wsl.localhost\Ubuntu-22.04\home\qile\ovss_runs\v6_bnfreeze_stage1\best_pareto.pth

## 本仓库包含的关键脚本快照（scripts/lseg/ref/）
- run_lseg_loveda_confmat.py：评测入口（LMDB + ckpt -> 指标 + confmat npz）
- finetune_lseg_loveda_head_v6.py：训练/微调 head（产出 v6 权重）
- gt_hist_range.py：分析不同 index 段的 GT 分布漂移
- summarize_npz_segments.py：汇总多段 npz 的指标

## 运行位置（很重要）
我们通常在 LandSegmenter repo 下运行评测脚本，因为原始脚本实际存放在那里：
/mnt/e/ovss_project/pythonProject1/repos/LandSegmenter/run_lseg_loveda_confmat.py

提示：如果你只想用本仓库的快照版本，也可以把 scripts/lseg/ref/run_lseg_loveda_confmat.py 复制回工作目录执行。

## 常见坑（必须写清）
- prompt_set / 类别列表 / no_bias / agg 必须与训练时一致；否则可能出现 missing/unexpected keys，结果不可信。
- stats_dir 必填：脚本会把 log/npz 写到该目录。

