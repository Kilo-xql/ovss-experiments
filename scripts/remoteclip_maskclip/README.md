# SAM3 + RemoteCLIP（MaskCLIP v2）复现：LoveDA val 0–200

本目录提供一个**可复现的零样本（zero-shot）baseline**：
- 用 **SAM3** 为每张图生成 proposal masks（候选区域）
- 对每个候选区域做 **bbox crop**，用 **RemoteCLIP** 做图文匹配分类得到语义标签
- 将“mask + 标签”回填合并成 7 类语义分割图，并计算 mIoU

> 说明：bbox crop 是这类 MaskCLIP / mask-then-CLIP 流程里最常见、最标准的做法。

---

## 环境
- conda 环境：`sam3`
- 需要加载 RemoteCLIP 相关环境变量：
  ```bash
  source /mnt/e/ovss_project/pythonProject1/scripts/env_clip.sh