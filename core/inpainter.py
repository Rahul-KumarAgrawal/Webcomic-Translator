"""
core/inpainter.py
Adapter around manga-image-translator for:
  - Speech bubble detection
  - Text region detection / OCR
  - Background inpainting (text removal)
  - Translated text rendering with custom font settings

This module wraps MIT's pipeline so the rest of the codebase stays
decoupled from MIT's internal API changes.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Force model downloads to D drive
os.environ.setdefault("MIT_MODELS_DIR", os.path.join(_ROOT, "models"))


@dataclass
class BubbleRegion:
    """A detected speech bubble / text region in a comic page."""
    x: int
    y: int
    w: int
    h: int
    source_text: str              # raw OCR text
    translated_text: str = ""     # filled in by translator
    font_cfg: dict = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def crop(self, image: Image.Image) -> Image.Image:
        """Return a cropped PIL Image of this region."""
        return image.crop(self.bbox)


class Inpainter:
    """
    Wraps manga-image-translator to detect bubbles, inpaint text,
    and redraw translated text with user-defined font settings.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._device = cfg.get("gpu_device", "cuda")
        self._mit_cfg = cfg.get("mit", {})

        # Lazy-loaded Koharu models (initialized on first use, freed via unload_models)
        self._yolo_model = None
        self._mocr_model = None
        self._models_dir = Path(_ROOT) / "Pipeline Koharu"

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_and_ocr(self, image_path: str) -> Tuple[Image.Image, List[BubbleRegion]]:
        """
        Main entry point: loads image, runs detection, then runs OCR on found regions.
        Returns the (possibly modified/upscaled) image and the list of BubbleRegions.
        """
        image = Image.open(image_path).convert("RGB")
        
        # DEBUG: Print initial image size
        logger.info(f"[DEBUG INPAINTER] Initial image.size = {image.size}")
        
        # ── 0. Global Pre-Process Upscale (Optional) ──────────────────
        global_upscale_impl = self.cfg.get("global_upscale_impl", "none").lower()
        if global_upscale_impl != "none" or self.cfg.get("global_upscale", False) is True:
            # Fallback
            if global_upscale_impl == "none" and self.cfg.get("global_upscale", False) is True:
                global_upscale_impl = "lanczos"
                
            w, h = image.size
            if global_upscale_impl == "waifu2x":
                logger.info(f"[Inpainter] Applying Waifu2x Global 2x Upscale: {w}x{h} -> {w*2}x{h*2}")
                try:
                    import asyncio
                    from manga_translator.upscaling.waifu2x import Waifu2xUpscaler
                    
                    async def _run_waifu2x():
                        import torch
                        upscaler = Waifu2xUpscaler()
                        if not upscaler.is_downloaded():
                            logger.info("[Inpainter] Downloading Waifu2x models/binaries...")
                            await upscaler.download()
                        
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        if not upscaler.is_loaded():
                            await upscaler.load(device=device)

                        logger.info(f"[Inpainter] Executing Waifu2x upscaler...")
                        up_images = await upscaler.upscale([image], 2.0)
                        return up_images[0]

                    image = asyncio.run(_run_waifu2x())
                    logger.info(f"[Inpainter] Waifu2x upscaling complete. New size: {image.size}")
                except Exception as e:
                    logger.warning(f"[Inpainter] Waifu2x failed: {e}. Falling back to Lanczos.")
                    image = image.resize((w*2, h*2), resample=Image.LANCZOS)
            else:
                logger.info(f"[Inpainter] Applying Lanczos Global 2x Upscale: {w}x{h} -> {w*2}x{h*2}")
                image = image.resize((w*2, h*2), resample=Image.LANCZOS)

        det_engine = self.cfg.get("detection_engine", "mit")
        ocr_engine = self.cfg.get("ocr_engine", "mit")
        logger.info("[Modular] Using detection=%s, ocr=%s", det_engine, ocr_engine)

        # ── 1. Detection ──────────────────────────────────────────────────
        if str(det_engine).lower() in ("yolo", "yolo_hybrid"):
            regions = self._run_yolo_detect(image)
        else:
            regions = self._run_mit_detect(image_path, image)

        # ── 1b. Split oversized regions (MIT only — YOLO is left untouched)
        if str(det_engine).lower() not in ("yolo", "yolo_hybrid"):
            regions = self._split_tall_regions(image, regions)

        # ── 1c. OCR Gap-Filling (Safety Net) ──────────────────────────
        # Enable if explicitly requested via engine or if global setting is ON
        is_hybrid = str(det_engine).lower() == "yolo_hybrid"
        if is_hybrid or self.cfg.get("enable_gap_filling", False):
            try:
                from core.paddleocr_wrapper import run_paddle_gap_filling
                new_data = run_paddle_gap_filling(image, regions, self.cfg)
                if new_data:
                    logger.info("[Modular] Gap-Filling found %d missed text areas.", len(new_data))
                    for d in new_data:
                        regions.append(BubbleRegion(
                            x=d["x"], y=d["y"], w=d["w"], h=d["h"],
                            source_text=d["text"]
                        ))
            except Exception as e:
                logger.warning(f"[Modular] Gap-Filling failed: {e}")

        # ── 2. OCR ────────────────────────────────────────────────────────
        if str(ocr_engine).lower() == "paddle":
            try:
                from core.paddleocr_wrapper import run_paddle_ocr_on_regions
                regions = run_paddle_ocr_on_regions(image, regions, self.cfg)
            except Exception as e:
                logger.error(f"Failed to run PaddleOCR, falling back: {e}")
        elif str(ocr_engine).lower() == "easyocr":
            try:
                from core.easyocr_wrapper import run_easyocr_on_regions
                regions = run_easyocr_on_regions(image, regions, self.cfg)
            except Exception as e:
                logger.error(f"Failed to run EasyOCR, falling back: {e}")
        elif str(ocr_engine).lower() == "tesseract":
            try:
                from core.tesseract_wrapper import run_tesseract_on_regions
                lang = self.cfg.get("source_lang", "eng_Latn")
                regions = run_tesseract_on_regions(image, regions, self.cfg)
            except Exception as e:
                logger.error(f"Failed to run Tesseract, falling back: {e}")
        elif str(ocr_engine).lower() in ("manga-ocr", "mit"):
            regions = self._run_manga_ocr_on_regions(image, regions)

        # ── 3. Final Deduplication & Merging ───────────────────────────
        regions = self._merge_nearby_regions(regions)

        logger.info(f"[DEBUG INPAINTER] Final image.size = {image.size}, Number of regions = {len(regions)}")
        for i, r in enumerate(regions):
            logger.debug(f"[DEBUG INPAINTER] Region {i}: bbox=({r.x}, {r.y}, ..., w={r.w}, h={r.h}), max_w={image.size[0]}, max_h={image.size[1]}")

        return image, regions

    # ── Koharu engine methods ─────────────────────────────────────────────────

    def _run_yolo_detect(self, image: Image.Image) -> List[BubbleRegion]:
        """Use YOLO detector from the Koharu pipeline for text region detection."""
        if not self._yolo_model:
            from ultralytics import YOLO
            model_path = self._models_dir / "Detection and Layout" / "comic-text-segmenter.pt"
            self._yolo_model = YOLO(str(model_path))
            logger.info("[Modular] YOLO text detector loaded.")

        # Run detection with configurable confidence
        conf = float(self.cfg.get("detection_confidence", 0.20))
        results = self._yolo_model(image, verbose=False, conf=conf, iou=0.45)
        regions = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                
                regions.append(BubbleRegion(
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    source_text=""
                ))
        return regions

    def _run_manga_ocr_on_regions(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Use Manga-OCR for high-quality Japanese text recognition."""
        if not self._mocr_model:
            from manga_ocr import MangaOcr
            model_path = self._models_dir / "OCR" / "manga-ocr-base"
            self._mocr_model = MangaOcr(str(model_path))
            logger.info("[Modular] MangaOCR loaded.")

        super_res = self.cfg.get("ocr_super_res", True)
        upscale_factor = float(self.cfg.get("ocr_upscale_factor", 2.0))

        for region in regions:
            try:
                crop = region.crop(image)
                
                if super_res:
                    # Upscale for better recognition on low-quality/fuzzy images
                    from PIL import ImageOps
                    w, h = crop.size
                    crop = crop.resize((int(w*upscale_factor), int(h*upscale_factor)), resample=Image.LANCZOS)
                    crop = ImageOps.autocontrast(crop.convert("L"), cutoff=2).convert("RGB")
                
                region.source_text = self._mocr_model(crop)
            except Exception as e:
                logger.warning("[MangaOCR] Failed on region %s: %s", region.bbox, e)
        return regions

    def unload_models(self):
        """Free VRAM by unloading lazy-loaded detection/OCR models."""
        if self._yolo_model:
            del self._yolo_model
            self._yolo_model = None
            logger.info("[Modular] YOLO model unloaded.")
        if self._mocr_model:
            del self._mocr_model
            self._mocr_model = None
            logger.info("[Modular] MangaOCR model unloaded.")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def inpaint(self, image: Image.Image, regions: List[BubbleRegion]) -> Image.Image:
        """
        Remove the original text from bubble regions (clean background).
        Returns a new PIL Image with text erased.
        """
        engine = self.cfg.get("inpaint_engine", "lama").lower()
        if engine == "aot":
            return self._run_aot_inpaint(image, regions)
        elif engine == "panelcleaner":
            return self._run_panelcleaner_inpaint(image, regions)
        elif engine == "solid":
            result = image.copy()
            for r in regions:
                self._clean_region(result, r)
            return result
        return self._run_mit_inpaint(image, regions)

    def _run_panelcleaner_inpaint(self, image: Image.Image, regions: List[BubbleRegion]) -> Image.Image:
        """
        Use PanelCleaner's LaMa model for inpainting.
        """
        try:
            import numpy as np
            from core.panelcleaner_wrapper import PanelCleanerPipeline
            
            # Create a full-page mask
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            for r in regions:
                mask[r.y:r.y+r.h, r.x:r.x+r.w] = 255

            if np.max(mask) == 0:
                return image

            np_img = np.array(image)
            pipeline = PanelCleanerPipeline(device=self._device)
            inpainted = pipeline.inpaint_lama(np_img, mask)
            
            logger.info("[PanelCleaner] Inpainting complete.")
            return Image.fromarray(inpainted)
        except Exception as exc:
            logger.warning("[PanelCleaner] Inpainting failed (%s). Falling back to MIT inpainting.", exc)
            return self._run_mit_inpaint(image, regions)

    def render_text(
        self,
        image: Image.Image,
        regions: List[BubbleRegion],
    ) -> Image.Image:
        """
        Draw translated text into each bubble region using the font settings
        in region.font_cfg (set by the caller before calling this method).

        Font size "auto" → text is scaled to fit the bubble bounds.
        Font color "auto" → white text on dark bubbles, black on light bubbles.
        """
        result = image.copy()
        draw = ImageDraw.Draw(result)

        for region in regions:
            if not region.translated_text:
                continue
            font_cfg = region.font_cfg
            color = font_cfg.get("color", "auto")
            size_setting = font_cfg.get("size", "auto")
            font_path = font_cfg.get("family", None)

            font_size = (
                self._auto_font_size(region, region.translated_text, font_path)
                if size_setting == "auto"
                else int(size_setting)
            )

            # Auto-detect text color based on bubble background brightness
            if color == "auto":
                bg_color = self._detect_bubble_bg_color(result, region)
                is_dark = self._is_dark_background(bg_color)
                color = "#FFFFFF" if is_dark else "#000000"

            font = self._load_font(font_path, font_size)
            self._draw_text_in_bubble(draw, region, region.translated_text, font, color)

        return result

    def process_page(
        self,
        image_path: str,
        translated_regions: List[BubbleRegion],
        output_path: str,
    ) -> str:
        """
        Full pipeline for one page:
          detect → inpaint → render → save.

        If translated_regions is pre-populated (caller did translation),
        only inpaint + render is run.
        Returns the output_path where the page was saved.
        """
        image = Image.open(image_path).convert("RGB")
        inpainted = self.inpaint(image, translated_regions)
        final = self.render_text(inpainted, translated_regions)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final.save(output_path, format="PNG", optimize=False)
        logger.debug("Page saved: %s", output_path)
        return output_path

    # ── Region splitting (oversized boxes) ─────────────────────────────────────

    def _split_tall_regions(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """
        Split regions that are abnormally tall (aspect ratio h/w > 3.0)
        by scanning for horizontal whitespace gaps inside the bounding box.
        This fixes the problem where the detector draws one giant box
        spanning multiple separate speech bubbles.
        """
        import numpy as np

        result = []
        for region in regions:
            aspect = region.h / max(region.w, 1)
            # Only attempt split on very tall regions
            if aspect < 3.0:
                result.append(region)
                continue

            # Crop the region from the image and convert to grayscale
            crop = image.crop((region.x, region.y,
                               region.x + region.w, region.y + region.h))
            gray = np.array(crop.convert("L"))

            # For each row, compute the std-dev of pixel values.
            # Rows that are "blank" (uniform background) will have very low std.
            row_std = np.std(gray, axis=1)

            # Also check if the row is mostly bright (white background)
            row_mean = np.mean(gray, axis=1)

            # A "gap row" is one that is fairly uniform AND bright
            # (typical manga bubble background is white/light)
            gap_mask = (row_std < 15) & (row_mean > 180)

            # Find contiguous runs of gap rows that are wide enough to be a real gap
            min_gap_height = max(8, int(region.h * 0.03))  # at least 3% of the region height
            splits = []
            in_gap = False
            gap_start = 0

            for i, is_gap in enumerate(gap_mask):
                if is_gap and not in_gap:
                    gap_start = i
                    in_gap = True
                elif not is_gap and in_gap:
                    gap_len = i - gap_start
                    if gap_len >= min_gap_height:
                        # Split at the midpoint of the gap
                        splits.append(gap_start + gap_len // 2)
                    in_gap = False

            if not splits:
                # No clear gaps found, keep the original region
                result.append(region)
                continue

            # Build sub-regions from the split points
            logger.info("[Split] Tall region (%dx%d, aspect=%.1f) split into %d parts",
                        region.w, region.h, aspect, len(splits) + 1)
            boundaries = [0] + splits + [region.h]
            for j in range(len(boundaries) - 1):
                sub_y = boundaries[j]
                sub_h = boundaries[j + 1] - sub_y
                if sub_h < 10:  # skip tiny slivers
                    continue
                result.append(BubbleRegion(
                    x=region.x,
                    y=region.y + sub_y,
                    w=region.w,
                    h=sub_h,
                    source_text="",  # OCR will fill this in later
                ))

        return result

    # ── Region merging ─────────────────────────────────────────────────────────

    def _merge_nearby_regions(self, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """
        Merge text regions that likely belong to the same speech bubble.

        MIT OCR often detects each text line as a separate region. This method
        groups regions that overlap horizontally and are close vertically
        (within a gap threshold), combining their bounding boxes and
        concatenating their OCR text so the full bubble text is translated
        as a single unit.
        """
        if len(regions) <= 1:
            return regions

        # Detect if text is vertical by checking region aspect ratios
        # Vertical CJK columns are taller than wide
        src_lang = (self.cfg.get("source_lang_override") or "").lower()
        is_cjk = any(k in src_lang for k in ("jp", "japan", "ko", "kor", "zh", "ch")) or src_lang == ""
        avg_aspect = sum(r.h / max(r.w, 1) for r in regions) / len(regions)
        is_vertical = is_cjk and avg_aspect > 1.5

        if is_vertical:
            # Vertical CJK text: columns read right-to-left
            regions = sorted(regions, key=lambda r: (-r.x, r.y))
        else:
            # Horizontal text: read left-to-right, top-to-bottom
            regions = sorted(regions, key=lambda r: (r.y, r.x))
        gap_threshold = 15  # max vertical pixel gap to consider same bubble

        merged = []
        current = regions[0]

        for next_r in regions[1:]:
            # Check overlap and gaps
            cur_x1, cur_x2 = current.x, current.x + current.w
            cur_y1, cur_y2 = current.y, current.y + current.h
            nxt_x1, nxt_x2 = next_r.x, next_r.x + next_r.w
            nxt_y1, nxt_y2 = next_r.y, next_r.y + next_r.h

            overlap_x = min(cur_x2, nxt_x2) - max(cur_x1, nxt_x1)
            overlap_y = min(cur_y2, nxt_y2) - max(cur_y1, nxt_y1)
            
            min_w = min(current.w, next_r.w)
            min_h = min(current.h, next_r.h)

            v_gap = next_r.y - cur_y2
            h_gap = next_r.x - cur_x2 if not is_vertical else cur_x1 - nxt_x2

            # Determine if we should merge
            should_merge = False
            
            if is_vertical:
                # For CJK vertical text: ONLY merge side-by-side columns (horizontal merge)
                # We want to merge if they overlap vertically and are close horizontally
                if overlap_y > 0.5 * min_h and abs(h_gap) < 25:
                    should_merge = True
                # Vertical stacking merge is DISABLED for CJK to prevent giant strips.
            else:
                # For horizontal text: rows are stacked (merging vertically)
                if overlap_x > 0.5 * min_w and v_gap < 15:
                    should_merge = True
                # Or side-by-side fragments
                elif overlap_y > 0.5 * min_h and abs(h_gap) < 10:
                    should_merge = True

            if should_merge:
                # Expand bounding box
                new_x1 = min(cur_x1, nxt_x1)
                new_y1 = min(cur_y1, nxt_y1)
                new_x2 = max(cur_x2, nxt_x2)
                new_y2 = max(cur_y2, nxt_y2)
                
                # Concatenate text (right-to-left for vertical, left-to-right for horizontal)
                if is_vertical and nxt_x1 > cur_x1:
                    combined_source = (next_r.source_text.strip() + " " + current.source_text.strip()).strip()
                    combined_trans  = (next_r.translated_text.strip() + " " + current.translated_text.strip()).strip()
                else:
                    combined_source = (current.source_text.strip() + " " + next_r.source_text.strip()).strip()
                    combined_trans  = (current.translated_text.strip() + " " + next_r.translated_text.strip()).strip()

                current = BubbleRegion(
                    x=new_x1, y=new_y1,
                    w=new_x2 - new_x1, h=new_y2 - new_y1,
                    source_text=combined_source,
                    translated_text=combined_trans
                )
            else:
                merged.append(current)
                current = next_r

        merged.append(current)
        return self._deduplicate_contained_regions(merged)

    def _deduplicate_contained_regions(self, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Remove larger regions that contain smaller, better-defined sub-regions."""
        if not regions:
            return []
            
        # Sort by area ascending so we check smaller bubbles first
        sorted_regions = sorted(regions, key=lambda r: r.w * r.h)
        remove_indices = set()
        
        for i, r_small in enumerate(sorted_regions):
            if i in remove_indices: continue
            
            for j, r_large in enumerate(sorted_regions):
                if i == j or j in remove_indices: continue
                if r_large.w * r_large.h <= r_small.w * r_small.h: continue
                
                # Check if r_small is inside r_large
                inter_x1 = max(r_small.x, r_large.x)
                inter_y1 = max(r_small.y, r_large.y)
                inter_x2 = min(r_small.x + r_small.w, r_large.x + r_large.w)
                inter_y2 = min(r_small.y + r_small.h, r_large.y + r_large.h)
                
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                small_area = r_small.w * r_small.h
                
                # If >80% of small bubble is inside large bubble, the large bubble 
                # is likely a redundant container. Remove it to allow better OCR on pieces.
                if inter_area > 0.8 * small_area:
                    remove_indices.add(j)
            
        keep = [r for idx, r in enumerate(sorted_regions) if idx not in remove_indices]
        # Return in original top-to-bottom order
        return sorted(keep, key=lambda r: (r.y, r.x))

    def _run_aot_inpaint(self, image: Image.Image, regions: List[BubbleRegion]) -> Image.Image:
        """
        Use AOT-GAN ONNX model from Pipeline Koharu for high-detail inpainting.
        Falls back to MIT inpainting if the model is not found or fails.
        """
        aot_model_path = self._models_dir / "Inpainting" / "aot-inpainting" / "aot.onnx"
        if not aot_model_path.exists():
            logger.warning("[AOT] Model not found at %s. Falling back to MIT inpainting.", aot_model_path)
            return self._run_mit_inpaint(image, regions)

        try:
            import onnxruntime as ort
            import numpy as np

            # Create a full-page mask
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            for r in regions:
                mask[r.y:r.y+r.h, r.x:r.x+r.w] = 255

            # If nothing to inpaint, return image
            if np.max(mask) == 0:
                return image

            # Convert to numpy arrays
            original = np.array(image)
            mask_3ch = np.stack([mask]*3, axis=-1) / 255.0

            # Preprocess image
            np_img = original.astype(np.float32) / 255.0
            
            h, w = np_img.shape[:2]
            
            # AOT-GAN requires dimensions to be multiples of 8
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                np_img = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            else:
                mask_padded = mask

            # AOT expects [1, 3, H, W] image and [1, 1, H, W] mask
            # Image is scaled to [-1, 1], Mask is scaled to [0, 1]
            img_tensor  = (np_img * 2.0 - 1.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            mask_tensor = (mask_padded / 255.0)[np.newaxis, np.newaxis].astype(np.float32)

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(str(aot_model_path), providers=providers)

            inp_name  = sess.get_inputs()[0].name
            mask_name = sess.get_inputs()[1].name
            out_name  = sess.get_outputs()[0].name

            # Run inference
            out_tensor = sess.run([out_name], {inp_name: img_tensor, mask_name: mask_tensor})[0]

            # Crop padding back off
            if pad_h > 0 or pad_w > 0:
                out_tensor = out_tensor[:, :, :h, :w]

            # Post-process [-1, 1] back to [0, 255]
            out = (out_tensor[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0
            out = np.clip(out, 0, 255)

            composite = (original * (1 - mask_3ch) + out * mask_3ch).astype(np.uint8)

            logger.info("[AOT] Inpainting complete via AOT-GAN ONNX.")
            return Image.fromarray(composite)

        except Exception as exc:
            logger.warning("[AOT] Inpainting failed (%s). Falling back to MIT inpainting.", exc)
            return self._run_mit_inpaint(image, regions)

    # ── MIT integration (detection + inpainting) ──────────────────────────────

    def _run_mit_detect(self, image_path: str, image: Image.Image) -> List[BubbleRegion]:
        """
        Use manga-image-translator's detection + OCR pipeline.
        Falls back to an empty list if MIT is not installed / not available.

        MIT uses an async API, so we bridge with asyncio.run().
        """
        try:
            import asyncio
            from manga_translator import MangaTranslator
            from manga_translator.config import (
                Config, Detector as MITDetector, Ocr as MITOcr,
                Renderer as MITRenderer, Inpainter as MITInpainter,
                Translator as MITTranslator,
            )
            from manga_translator.utils import load_image

            # Build MangaTranslator with proper params dict
            params = {
                "use_gpu": self._device.startswith("cuda"),
                "verbose": False,
                "kernel_size": 3,
                "pre_dict": None,
                "post_dict": None,
            }
            translator_obj = MangaTranslator(params)

            # Build config — only detect + OCR, skip translation/rendering
            cfg = Config()
            det_key = self._mit_cfg.get("detector", "default")
            ocr_key = self._mit_cfg.get("ocr", "48px")
            try:
                cfg.detector.detector = MITDetector(det_key)
            except (ValueError, KeyError):
                cfg.detector.detector = MITDetector.default
            try:
                cfg.ocr.ocr = MITOcr(ocr_key)
            except (ValueError, KeyError):
                cfg.ocr.ocr = MITOcr.ocr48px

            # Use 'none' for translator/renderer/inpainter — we only want detect+OCR
            cfg.translator.translator = MITTranslator.none
            cfg.render.renderer = MITRenderer.none
            cfg.inpainter.inpainter = MITInpainter.none

            # Run the translate pipeline (which does detect → OCR → textline merge)
            async def _detect():
                ctx = await translator_obj.translate(image, cfg, skip_context_save=True)
                return ctx

            ctx = asyncio.run(_detect())

            # Extract text regions from context
            regions = []
            text_regions = getattr(ctx, "text_regions", None) or []
            for tr in text_regions:
                # text_regions have .xyxy (x1,y1,x2,y2) or .aabb attributes
                xyxy = getattr(tr, "xyxy", None)
                if xyxy is None:
                    aabb = getattr(tr, "aabb", None)
                    if aabb is not None:
                        xyxy = aabb
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ocr_text = getattr(tr, "text", "") or ""
                regions.append(BubbleRegion(
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    source_text=ocr_text,
                ))

            # Merge nearby regions that belong to the same speech bubble
            regions = self._merge_nearby_regions(regions)
            logger.debug("Detected %d text regions via MIT (after merging).", len(regions))
            return regions

        except ImportError:
            logger.error("manga-image-translator is not installed. Run setup.bat to install.")
            raise
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            logger.error("MIT detect failed (%s)\n%s", exc, tb)
            raise

    def _run_mit_inpaint(
        self, image: Image.Image, regions: List[BubbleRegion]
    ) -> Image.Image:
        """
        Inpaint (erase) text from bubble regions.
        Falls back to white-fill if MIT inpainting is not available.
        """
        try:
            import asyncio
            from manga_translator.config import Inpainter as MITInpainter, InpainterConfig
            from manga_translator.inpainting import dispatch as dispatch_inpainting

            inpainter_key = self._mit_cfg.get("inpainter", "default")
            try:
                inp_enum = MITInpainter(inpainter_key)
            except (ValueError, KeyError):
                inp_enum = MITInpainter.default

            np_img = np.array(image)

            use_sam = self.cfg.get("use_sam_masks", False)
            if use_sam:
                from core.sam_wrapper import generate_bubble_mask_from_regions, unload_sam
                mask = generate_bubble_mask_from_regions(np_img, regions)

                # Always free SAM before LaMa runs to avoid dtype / VRAM conflicts.
                # For cloud-API engines SAM will be re-loaded lazily on the next page;
                # for local engines (nllb, ollama) it stays unloaded to save VRAM.
                unload_sam()
            else:
                mask = np.zeros(np_img.shape[:2], dtype=np.uint8)
                for r in regions:
                    x1, y1, x2, y2 = r.bbox
                    mask[y1:y2, x1:x2] = 255

            inp_cfg = InpainterConfig(inpainter=inp_enum)

            async def _inpaint():
                return await dispatch_inpainting(
                    inp_enum, np_img, mask, inp_cfg,
                    inpainting_size=inp_cfg.inpainting_size,
                    device=self._device, verbose=False,
                )

            cleaned = asyncio.run(_inpaint())
            return Image.fromarray(cleaned)

        except Exception as exc:
            logger.warning("MIT inpaint failed (%s), using smart-fill fallback.", exc)
            result = image.copy()
            for r in regions:
                self._clean_region(result, r)
            return result

    def _clean_region(self, image: Image.Image, region: "BubbleRegion") -> None:
        """
        Clean a text region by filling with the sampled background color
        and blending edges with a Gaussian blur for a seamless result.
        """
        x1, y1, x2, y2 = region.bbox
        img_w, img_h = image.size

        # Add padding around the text region to cover stroke edges
        pad = 4
        fx1 = max(0, x1 - pad)
        fy1 = max(0, y1 - pad)
        fx2 = min(img_w, x2 + pad)
        fy2 = min(img_h, y2 + pad)

        # Sample background color from the edges of the region
        bg_color = self._sample_bg_color(image, region)

        # Create a patch filled with the background color
        patch_w = fx2 - fx1
        patch_h = fy2 - fy1
        if patch_w <= 0 or patch_h <= 0:
            return

        # Fill the region with the sampled background color
        fill_patch = Image.new("RGB", (patch_w, patch_h), bg_color)

        # Create a soft-edged mask for blending (feathered edges)
        mask = Image.new("L", (patch_w, patch_h), 255)
        mask_draw = ImageDraw.Draw(mask)
        # Make edges transparent (feather = 6px)
        feather = 6
        for i in range(feather):
            alpha = int(255 * (i / feather))
            # Top edge
            mask_draw.rectangle([i, i, patch_w - 1 - i, i], fill=alpha)
            # Bottom edge
            mask_draw.rectangle([i, patch_h - 1 - i, patch_w - 1 - i, patch_h - 1 - i], fill=alpha)
            # Left edge
            mask_draw.rectangle([i, i, i, patch_h - 1 - i], fill=alpha)
            # Right edge
            mask_draw.rectangle([patch_w - 1 - i, i, patch_w - 1 - i, patch_h - 1 - i], fill=alpha)

        # Blur the mask slightly for smoother blending
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

        # Paste the filled patch using the feathered mask
        image.paste(fill_patch, (fx1, fy1), mask)

    def _sample_bg_color(self, image: Image.Image, region: "BubbleRegion") -> tuple:
        """
        Sample the dominant background color of a bubble region
        by looking at edge pixels (which are less likely to contain text).
        Returns an (R, G, B) tuple.
        """
        x1, y1, x2, y2 = region.bbox
        img_w, img_h = image.size

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if x2 - x1 < 4 or y2 - y1 < 4:
            # For very small regions, sample center pixel instead of assuming white
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = min(max(cx, 0), img_w - 1)
            cy = min(max(cy, 0), img_h - 1)
            return tuple(image.getpixel((cx, cy)))[:3]

        np_img = np.array(image)

        # Collect pixels from the edges of the bounding box (2px border)
        edge_pixels = []
        border = 2

        # Top edge
        if y1 + border <= y2:
            edge_pixels.append(np_img[y1:y1 + border, x1:x2])
        # Bottom edge
        if y2 - border >= y1:
            edge_pixels.append(np_img[y2 - border:y2, x1:x2])
        # Left edge
        if x1 + border <= x2:
            edge_pixels.append(np_img[y1:y2, x1:x1 + border])
        # Right edge
        if x2 - border >= x1:
            edge_pixels.append(np_img[y1:y2, x2 - border:x2])

        if not edge_pixels:
            # Fallback: sample center pixel
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return tuple(image.getpixel((cx, cy)))[:3]

        # Stack all edge pixels and compute median color (robust to outliers/text)
        all_pixels = np.concatenate([p.reshape(-1, 3) for p in edge_pixels], axis=0)
        median_color = np.median(all_pixels, axis=0).astype(int)

        return tuple(median_color)

    def _detect_bubble_bg_color(self, image: Image.Image, region: "BubbleRegion") -> tuple:
        """
        Detect the background color of a bubble region on the (already inpainted) image.
        Uses the center area of the bubble to avoid edge artifacts.
        Returns an (R, G, B) tuple.
        """
        x1, y1, x2, y2 = region.bbox
        img_w, img_h = image.size

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        w = x2 - x1
        h = y2 - y1
        if w < 2 or h < 2:
            return (255, 255, 255)

        # Sample from the inner 60% of the bubble to avoid border/edge artifacts
        margin_x = max(1, int(w * 0.2))
        margin_y = max(1, int(h * 0.2))
        inner_x1 = x1 + margin_x
        inner_y1 = y1 + margin_y
        inner_x2 = x2 - margin_x
        inner_y2 = y2 - margin_y

        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            # Region too small for margin, just sample center pixel
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return tuple(image.getpixel((cx, cy)))[:3]

        np_img = np.array(image)
        inner_pixels = np_img[inner_y1:inner_y2, inner_x1:inner_x2].reshape(-1, 3)
        median_color = np.median(inner_pixels, axis=0).astype(int)
        return tuple(median_color)

    @staticmethod
    def _is_dark_background(bg_color: tuple) -> bool:
        """
        Determine if a background color is dark using luminance.
        Uses the standard perceptual luminance formula.
        Returns True if the background is dark (text should be white).
        """
        r, g, b = bg_color[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance < 128

    # ── Font / text rendering helpers ─────────────────────────────────────────

    def _load_font(self, font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        # 1. User-configured font
        try:
            if font_path and Path(font_path).exists():
                return ImageFont.truetype(str(font_path), size)
        except Exception:
            pass
        # 2. Common system fonts (Windows / Linux)
        _SYSTEM_FONTS = [
            # Windows
            "C:/Windows/Fonts/Arial.ttf",
            "C:/Windows/Fonts/Calibri.ttf",
            "C:/Windows/Fonts/Verdana.ttf",
            "C:/Windows/Fonts/Tahoma.ttf",
            # Linux (WSL / Ubuntu)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for fp in _SYSTEM_FONTS:
            try:
                if Path(fp).exists():
                    return ImageFont.truetype(fp, size)
            except Exception:
                continue
        # 3. Last resort: PIL default (fixed 11px)
        return ImageFont.load_default()

    def _auto_font_size(
        self, region: BubbleRegion, text: str, font_path: Optional[str]
    ) -> int:
        """Binary-search for the largest font size where ALL of these hold:

        1. Every individual word fits on one line (no mid-word breaks).
        2. The wrapped lines collectively fit within the bubble height.
        """
        padding = 6
        max_w = max(region.w - padding * 2, 10)
        max_h = max(region.h - padding * 2, 10)

        words = text.split() or [text]

        lo, hi = 8, 48   # cap at 48 — above that text looks comically large
        best = lo

        # Temp image just for measuring text size
        tmp_img  = Image.new("RGB", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)

        while lo <= hi:
            mid  = (lo + hi) // 2
            font = self._load_font(font_path, mid)

            # ── Constraint 1: every word must fit on its own line ──────────
            emoji_font = self._load_emoji_font(mid)
            longest_word_w = max(
                (self._get_mixed_textlength(w, font, emoji_font, tmp_draw) for w in words), default=0
            )
            if longest_word_w > max_w:
                hi = mid - 1
                continue

            # ── Constraint 2: all wrapped lines must fit in height ─────────
            # Wrap text using mixed length calculation
            wrap_lines: List[str] = []
            current = ""
            for word in words:
                test = (current + " " + word).strip()
                if self._get_mixed_textlength(test, font, emoji_font, tmp_draw) <= max_w:
                    current = test
                else:
                    if current: wrap_lines.append(current)
                    current = word
            if current: wrap_lines.append(current)

            line_h  = mid + 2
            total_h = len(wrap_lines) * line_h
            if total_h <= max_h:
                best = mid
                lo   = mid + 1
            else:
                hi = mid - 1

        return best

    def _wrap_text(
        self, text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw
    ) -> List[str]:
        """Word-wrap text to fit within max_w pixels.

        Words are NEVER broken mid-character.  _auto_font_size() guarantees
        that the font was chosen small enough for every word to fit on one line,
        so each word will always find room on a new line at worst.
        """
        words   = text.split()
        lines: List[str] = []
        current = ""

        for word in words:
            test = (current + " " + word).strip()
            if draw.textlength(test, font=font) <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word   # start a new line with this word

        if current:
            lines.append(current)
        return lines or [text]

    def _load_emoji_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a system emoji font as a fallback."""
        # Project root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Priority fallback list
        fallbacks = [
            os.path.join(root, "fonts", "Arial-Unicode-Regular.ttf"), # Local priority
            "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
            "C:/Windows/Fonts/symbola.ttf",   # Symbola
            "C:/Windows/Fonts/arialuni.ttf",  # Arial Unicode MS
            "C:/Windows/Fonts/msgothic.ttc",  # MS Gothic
            "seguiemj.ttf",
            "symbola.ttf",
            "arialuni.ttf"
        ]
        for f in fallbacks:
            try:
                if os.path.exists(f) or not os.path.isabs(f):
                    return ImageFont.truetype(f, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _is_emoji(self, char: str) -> bool:
        """Detect if a character is a heart or common manga symbol."""
        # Expanded list of hearts and common manga symbols
        if char in "♥♡❤💔❣🖤💕💞💓💗💖💘💝✨💢⭐🌟💫💨💦💧🔥💨💤":
            return True
        # Symbols and Dingbats ranges (includes many hearts)
        code = ord(char)
        if 0x2600 <= code <= 0x26FF: # Miscellaneous Symbols
            return True
        if 0x2700 <= code <= 0x27BF: # Dingbats
            return True
        # Basic emoji range
        return code > 0x2000

    def _draw_mixed_line(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        line: str,
        primary_font: ImageFont.FreeTypeFont,
        emoji_font: ImageFont.FreeTypeFont,
        color: str,
        outline_color: str
    ) -> None:
        """Draw a line of text, switching fonts for emoji/symbols."""
        current_x = x
        for char in line:
            # Determine if we should use the emoji fallback
            use_emoji = self._is_emoji(char)
            
            # Additional check: if primary font doesn't have the glyph, use emoji font
            if not use_emoji:
                try:
                    # getmask().getbbox() is None if the glyph is missing
                    if primary_font.getmask(char).getbbox() is None:
                        use_emoji = True
                except Exception:
                    use_emoji = True

            font = emoji_font if use_emoji else primary_font
            
            # Draw outline
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((current_x + dx, y + dy), char, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((current_x, y), char, font=font, fill=color)
            
            # Advance X
            current_x += int(draw.textlength(char, font=font))

    def _get_mixed_textlength(self, text: str, primary_font: ImageFont.FreeTypeFont, emoji_font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw) -> float:
        """Calculate length of a string using multiple fonts."""
        total = 0.0
        for char in text:
            use_emoji = self._is_emoji(char)
            if not use_emoji:
                try:
                    if primary_font.getmask(char).getbbox() is None:
                        use_emoji = True
                except Exception:
                    use_emoji = True
            font = emoji_font if use_emoji else primary_font
            total += draw.textlength(char, font=font)
        return total

    def _draw_text_in_bubble(
        self,
        draw: ImageDraw.ImageDraw,
        region: BubbleRegion,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: str,
    ) -> None:
        """Draw word-wrapped, centred text inside a bubble region with emoji support."""
        padding   = 6
        max_w     = max(region.w - padding * 2, 10)
        
        # Load emoji fallback at same size
        font_size = font.size if hasattr(font, "size") else 12
        emoji_font = self._load_emoji_font(font_size)

        # Wrap text using mixed length calculation
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if self._get_mixed_textlength(test, font, emoji_font, draw) <= max_w:
                current = test
            else:
                if current: lines.append(current)
                current = word
        if current: lines.append(current)
        
        line_h    = font_size + 2
        total_h   = len(lines) * line_h

        start_y       = region.y + (region.h - total_h) // 2
        bubble_left   = region.x + padding
        bubble_right  = region.x + region.w - padding
        outline_color = "#000000" if color.upper() in ("#FFFFFF", "#FFF", "WHITE") else "#FFFFFF"

        for i, line in enumerate(lines):
            y = start_y + i * line_h
            line_w = int(self._get_mixed_textlength(line, font, emoji_font, draw))
            
            x = region.x + (region.w - line_w) // 2
            x = max(bubble_left, min(x, bubble_right - line_w))

            self._draw_mixed_line(draw, x, y, line, font, emoji_font, color, outline_color)
