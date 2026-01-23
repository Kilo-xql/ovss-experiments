# RemoteCLIP + SAM3 proposals（MaskCLIP v2 baseline）— LoveDA val 0–200

本目录复现一个**零样本（zero-shot）语义分割 baseline**：

**SAM3 生成 proposals masks → 对每个 mask 做 bbox crop → RemoteCLIP 分类 → 回填合成 7 类分割图 → 计算 mIoU**

> 说明：SAM3 在这里**只负责切 proposal（候选区域）**；语义分类主要由 **RemoteCLIP** 完成。  
> bbox crop 是 mask-then-CLIP/MaskCLIP 风格里最常见的实现方式。

---

## 需要准备的 3 个输入（本地已有即可）

1) LoveDA val LMDB  
`/mnt/e/ovss_project/pythonProject1/data/test/loveda_p512_val.lmdb`

2) SAM3 proposals（每张图一个 npz，形如 `00000.npz`）  
`/mnt/e/ovss_project/pythonProject1/repos/clip/runs/val_0_200/masks_npz/`

3) RemoteCLIP 权重（不提交到 GitHub）  
例如（ViT-B-32）：  
`/mnt/e/ovss_project/pythonProject1/repos/clip/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`

---

## 环境

```bash
conda activate sam3
source /mnt/e/ovss_project/pythonProject1/scripts/env_clip.sh
source /mnt/e/ovss_project/pythonProject1/scripts/env_clip.sh
