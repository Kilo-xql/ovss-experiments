# SAM3 – LoveDA (LMDB) Eval/Vis

目标：把 SAM3 的评测/可视化链路写清楚（路径明确、命令短），方便以后换数据集复用。

## 本机关键路径（WSL）
- sam3_code（Python 包 sam3.*）：/mnt/e/ovss_project/pythonProject1/repos/sam3_code
- 权重 sam3.pt：/mnt/e/ovss_project/pythonProject1/repos/SAM3/sam3.pt
- LoveDA val LMDB：/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb
- 输出（eval 0-200）：/mnt/e/ovss_project/pythonProject1/runs/sam3_loveda/small_0_200
- 输出（vis）：/mnt/e/ovss_project/pythonProject1/runs/sam3_loveda/vis/

## 环境与依赖
建议 conda env：sam3
常见缺包：einops / decord / pycocotools
安装：pip install einops decord pycocotools

## 本仓库提供的 runner（短命令）
- Eval 0-200：
  bash scripts/sam3/run/run_eval_0_200.sh
- 可视化单张（默认 idx=0，可传 idx）：
  bash scripts/sam3/run/run_vis_one.sh
  bash scripts/sam3/run/run_vis_one.sh 123

## 换数据集怎么改
1) 换 LMDB：改 runner 顶部的 LMDB=...
2) 新 LMDB 长度不同：改 eval 的 start/end
3) 类别不同：需要改 eval/vis 脚本里的 prompt 列表 / label 映射
4) BGR/RGB：若颜色怪或预测异常但不报错，优先检查图像通道是否需要 BGR->RGB（img = img[..., ::-1]）

## 常见排错
- ModuleNotFoundError: sam3
  说明 PYTHONPATH 没包含 sam3_code，可先执行：
  export PYTHONPATH=/mnt/e/ovss_project/pythonProject1/repos/sam3_code:$PYTHONPATH
