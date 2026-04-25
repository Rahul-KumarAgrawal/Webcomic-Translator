import os
import urllib.request
import logging
import cv2
import numpy as np
import torch
from typing import List

logger = logging.getLogger(__name__)

# Global state for lazy-loading
_sam_predictor = None
_sam_model = None

def get_sam_checkpoint_path() -> str:
    """Ensure the SAM vit_b model is downloaded and return its path."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, "model", "cache")
    os.makedirs(model_dir, exist_ok=True)

    ckpt_path = os.path.join(model_dir, "sam_vit_b_01ec64.pth")
    if not os.path.exists(ckpt_path):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        logger.info(f"Downloading SAM vit_b checkpoint to {ckpt_path} ...")
        try:
            urllib.request.urlretrieve(url, ckpt_path)
            logger.info("SAM checkpoint downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download SAM checkpoint: {e}")
            raise

    return ckpt_path

def get_sam_predictor():
    """Lazy load the SAM model and predictor.  Kept in float32 to avoid
    dtype conflicts with downstream models (LaMa inpainter)."""
    global _sam_predictor, _sam_model

    if _sam_predictor is None:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything is not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        ckpt_path = get_sam_checkpoint_path()
        logger.info(f"Loading SAM model from {ckpt_path} ...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam_model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
        _sam_model = _sam_model.to(device)
        # NOTE: intentionally NOT calling .half() — LaMa autocast breaks
        # when SAM leaves fp16 state on the CUDA context.

        _sam_predictor = SamPredictor(_sam_model)
        logger.info("SAM loaded into VRAM (float32).")

    return _sam_predictor

def unload_sam():
    """Free SAM from VRAM entirely."""
    global _sam_predictor, _sam_model
    if _sam_predictor is not None or _sam_model is not None:
        logger.info("Unloading SAM from VRAM...")
        _sam_predictor = None
        _sam_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def sam_from_boxes(predictor, boxes: List[List[int]]):
    """Per-box SAM prediction → list of boolean (H, W) masks."""
    masks = []
    for box in boxes:
        box_np = np.array(box)
        mask, _, _ = predictor.predict(
            box=box_np,
            multimask_output=False,
        )
        masks.append(mask[0])
    return masks

def refine_masks(mask_list, image_shape) -> np.ndarray:
    """Merge boolean masks → dilated + smoothed uint8 mask (0 / 255)."""
    if not mask_list:
        return np.zeros(image_shape[:2], dtype=np.uint8)

    final_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for m in mask_list:
        final_mask = np.maximum(final_mask, m.astype(np.uint8) * 255)

    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    final_mask = cv2.medianBlur(final_mask, 5)

    return final_mask

def generate_bubble_mask_from_regions(image_np: np.ndarray, regions: list) -> np.ndarray:
    """Generate a pixel-perfect bubble mask using SAM.

    Returns a uint8 mask (0 / 255) the same size as image_np.
    """
    if not regions:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    predictor = get_sam_predictor()
    predictor.set_image(image_np)

    boxes = [list(r.bbox) for r in regions]
    sam_masks = sam_from_boxes(predictor, boxes)
    final_mask = refine_masks(sam_masks, image_np.shape)

    # Free the large per-image embedding immediately
    predictor.reset_image()

    return final_mask
