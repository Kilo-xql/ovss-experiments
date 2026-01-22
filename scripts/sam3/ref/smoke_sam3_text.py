import os
import numpy as np
from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CKPT = "/mnt/e/ovss_project/pythonProject1/repos/SAM3/sam3.pt"
IMG  = os.environ.get("SAM3_SMOKE_IMG", "")

def main():
    assert os.path.isfile(CKPT), f"ckpt not found: {CKPT}"
    assert IMG and os.path.isfile(IMG), "please set SAM3_SMOKE_IMG to an existing image path"

    # 关键：尝试用本地 ckpt 加载
    try:
        model = build_sam3_image_model(checkpoint=CKPT)
    except TypeError:
        model = build_sam3_image_model(checkpoint_path=CKPT)

    processor = Sam3Processor(model)

    image = Image.open(IMG).convert("RGB")
    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt="road")

    masks = out["masks"]
    scores = out["scores"]

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    print("num_masks:", len(masks))
    if len(masks) == 0:
        print("No masks predicted.")
        return

    best = masks[scores.argmax()]

    # best 可能是 (H,W) 或 (1,H,W) 或 (H,W,1)，统一压成 (H,W)
    best = np.asarray(best)
    if best.ndim == 3:
        # (1,H,W) 或 (H,W,1)
        if best.shape[0] == 1:
            best = best[0]
        elif best.shape[-1] == 1:
            best = best[..., 0]

    best = (best.astype(np.uint8) * 255)

    save_path = "/mnt/e/ovss_project/pythonProject1/runs/sam3_loveda/smoke_best_mask.png"
    Image.fromarray(best).save(save_path)

    print("saved:", save_path)

if __name__ == "__main__":
    main()
