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
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageFilter
from typing import List, Tuple, Optional, Dict
import numpy as np
from pathlib import Path

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
    confidence: float = 1.0       # OCR confidence (0.0 - 1.0)
    font_cfg: dict = field(default_factory=dict)
    bubble_id: int = -1           # -1 if outside any bubble (narration), otherwise ID of the bubble mask
    mask_pts: Optional[np.ndarray] = None  # Full segmentation mask points from YOLO
    skip_inpaint: bool = False    # True if the bubble should be ignored (keep original art)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def crop(self, image: Image.Image) -> Image.Image:
        """Return a cropped PIL Image of this region."""
        return image.crop(self.bbox)


import torch
from typing import List, Optional
from PIL import Image
import numpy as np

class MayoSegmenter:
    """Specialized loader for Mayocream .safetensors models"""
    def __init__(self, model_path: str, architecture: str = 'unet', device: str = 'cuda'):
        import segmentation_models_pytorch as smp
        from safetensors.torch import load_file
        
        self.device = device
        # unet for bubbles, unet (or dbnet) for text
        if architecture == 'unet':
            self.model = smp.Unet(
                encoder_name="efficientnet-b3",
                encoder_weights=None,
                in_channels=3,
                classes=1,
            ).to(device)
        else:
            # Fallback to a standard Unet if unknown
            self.model = smp.Unet(encoder_name="efficientnet-b3", classes=1).to(device)
            
        weights = load_file(model_path)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()

    def predict(self, image: Image.Image, conf=0.5):
        img_w, img_h = image.size
        img = image.resize((512, 512)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Standard Normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device).to(torch.float32)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        import cv2
        mask_full = cv2.resize(mask, (img_w, img_h))
        return mask_full > conf

class Inpainter:
    """
    Wraps manga-image-translator to detect bubbles, inpaint text,
    and redraw translated text with user-defined font settings.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Detect device
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Inpainter] Using device: {self.device}")
        
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

        det_engine = self.cfg.get("detection_engine", "default")
        ocr_engine = self.cfg.get("ocr_engine", "default")
        inp_engine = self.cfg.get("inpaint_engine", "lama")
        logger.info("[Modular] Using detection=%s, ocr=%s, inpaint=%s", det_engine, ocr_engine, inp_engine)

        # ── 1. Detection ──────────────────────────────────────────────────
        if str(det_engine).lower() in ("yolo", "yolo_hybrid"):
            regions = self._run_yolo_detect(image)
        elif str(det_engine).lower() == "ogkalu_combine":
            regions = self._run_ogkalu_combine_detect(image)
        elif str(det_engine).lower() == "ogkalu_stable_dual":
            regions = self._run_ogkalu_stable_dual_detect(image)
        elif str(det_engine).lower() == "mayo_github":
            regions = self._run_mayo_github_detect(image)
        else:
            regions = self._run_mit_detect(image_path, image)

        # ── 1b. Split oversized regions (Disabled to prevent vertical text fragmentation)
        # if str(det_engine).lower() not in ("yolo", "yolo_hybrid"):
        #     regions = self._split_tall_regions(image, regions)
        pass

        # ── 1c. OCR Gap-Filling (Safety Net) ──────────────────────────
        # Only run if explicitly enabled in settings.
        if self.cfg.get("enable_gap_filling", False):
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
                # Use a temporary flag on self to only warn once per session
                if not hasattr(self, "_gap_fill_warned"):
                    logger.warning(f"[Modular] Gap-Filling failed (will only warn once): {e}")
                    self._gap_fill_warned = True
        else:
            if not hasattr(self, "_gap_fill_off_logged"):
                logger.info("[Modular] Gap-Filling is OFF (skipping missed text scan).")
                self._gap_fill_off_logged = True

        # ── 2. OCR ────────────────────────────────────────────────────────
        if str(ocr_engine).lower() in ("paddle", "paddle_vertical"):
            try:
                from core.paddleocr_wrapper import run_paddle_ocr_on_regions
                force_vert = str(ocr_engine).lower() == "paddle_vertical"
                regions = run_paddle_ocr_on_regions(image, regions, self.cfg, force_vertical=force_vert)
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
        elif str(ocr_engine).lower() == "pororo":
            try:
                regions = self._run_pororo_ocr(image, regions)
            except Exception as e:
                print("\n" + "!"*60)
                print(f"⚠️  WARNING: Pororo failed: {e}")
                print("Falling back to MIT Mayo (JPN)...")
                print("!"*60 + "\n")
                regions = self._run_manga_ocr_on_regions(image, regions)
        elif str(ocr_engine).lower() == "ppocr-v5":
            try:
                regions = self._run_ppocr_v5_ocr(image, regions)
            except Exception as e:
                print("\n" + "!"*60)
                print(f"⚠️  WARNING: PPOCR-v5 failed! Error: {e}. Falling back to MIT Mayo (JPN)...")
                print("!"*60 + "\n")
                regions = self._run_manga_ocr_on_regions(image, regions)
        elif str(ocr_engine).lower() == "manga-ocr":
            regions = self._run_manga_ocr_on_regions(image, regions)
        elif str(ocr_engine).lower() == "ogkalu_ocr":
            try:
                regions = self._run_ogkalu_ocr_onnx(image, regions)
            except Exception as e:
                logger.warning(f"[Ogkalu OCR] Failed: {e}. Falling back to Manga-OCR.")
                regions = self._run_manga_ocr_on_regions(image, regions)
        elif str(ocr_engine).lower() == "mit":
            # MIT Mayo (JPN) fallback/direct
            regions = self._run_mit_ocr_on_regions(image, regions)

        # ── 2b. Symbol-Only Fallback for Empty OCR Results ────────────
        regions = self._fallback_symbol_ocr(image, regions, str(ocr_engine).lower())

        # ── 3. Font Style Detection ─────────────────────────────────────
        # NOTE: Yuzumarker runs at RENDER time (inpaint/render phase), not here.
        # Running it here per-bubble during OCR is too early and wastes VRAM.

        # ── 3. Final Deduplication & Merging ───────────────────────────
        regions = self._merge_nearby_regions(regions)

        # ── 4. SFX / Noise Filtering ───────────────────────────────────
        regions = self._apply_sfx_strictness_filter(image, regions)

        # ── 5. Nuisance (Tiny Free-Floating) Filtering ─────────────────
        regions = self._apply_nuisance_filter(regions)
        


        logger.info(f"[DEBUG INPAINTER] Final image.size = {image.size}, Number of regions = {len(regions)}")
        for i, r in enumerate(regions):
            
            # Universal CJK De-fragmenter (Applies to all OCR engines including Pororo)
            if r.source_text:
                import re
                # This explicitly looks for spacing between CJK characters and removes it
                r.source_text = re.sub(r'([\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff])\s+([\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff])', r'\1\2', r.source_text)
                r.source_text = re.sub(r'([\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff])\s+([\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff])', r'\1\2', r.source_text)
                r.source_text = re.sub(r'\s+', ' ', r.source_text).strip()

            logger.debug(f"[DEBUG INPAINTER] Region {i}: bbox=({r.x}, {r.y}, ..., w={r.w}, h={r.h}), max_w={image.size[0]}, max_h={image.size[1]}")

        return image, regions

    def _fallback_symbol_ocr(self, image: Image.Image, regions: List[BubbleRegion], ocr_engine: str) -> List[BubbleRegion]:
        """
        If a bubble is detected (e.g. YOLO) and OCR returned empty,
        perform a high-contrast crop and use ANY available OCR engine 
        to look for punctuation (!, ?, ., ~, etc.).
        """
        import re
        from PIL import ImageOps

        # Only process regions with empty source_text
        for r in regions:
            if r.source_text and r.source_text.strip():
                continue
            
            x1, y1, x2, y2 = r.bbox
            pad = 15
            w_orig, h_orig = image.size
            crop_rect = (max(0, x1-pad), max(0, y1-pad), min(w_orig, x2+pad), min(h_orig, y2+pad))
            crop_pil = image.crop(crop_rect).convert("L")
            
            crop_pil = ImageOps.autocontrast(crop_pil, cutoff=0)

            w, h = crop_pil.size
            if w > 0 and h > 0:
                crop_pil = crop_pil.resize((w * 4, h * 4), resample=Image.LANCZOS)
                
            text = ""
            
            # Helper to run Paddle
            def try_paddle():
                try:
                    from core.paddleocr_wrapper import get_paddle_ocr
                    import numpy as np
                    ocr = get_paddle_ocr("en")
                    if ocr:
                        crop_np = np.array(crop_pil.convert("RGB"))
                        # Switch BGR for Paddle
                        crop_np = crop_np[:, :, ::-1]
                        result = ocr.ocr(crop_np, cls=False)
                        if result and result[0]:
                            return " ".join([line[1][0] for line in result[0] if line])
                except Exception as e:
                    logger.debug(f"Paddle fallback failed: {e}")
                return ""

            # Helper to run Tesseract
            def try_tesseract():
                try:
                    import pytesseract
                    from core.tesseract_wrapper import tesseract_path
                    if tesseract_path:
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                        config = '--psm 11'
                        return pytesseract.image_to_string(crop_pil, lang='eng', config=config).strip()
                except Exception as e:
                    logger.debug(f"Tesseract fallback failed: {e}")
                return ""

            # Helper to run EasyOCR
            def try_easyocr():
                try:
                    from core.easyocr_wrapper import get_easyocr_reader
                    import numpy as np
                    reader = get_easyocr_reader(["en"])
                    if reader:
                        crop_np = np.array(crop_pil.convert("RGB"))
                        results = reader.readtext(crop_np, detail=0, paragraph=False)
                        if results:
                            return " ".join(results)
                except Exception as e:
                    logger.debug(f"EasyOCR fallback failed: {e}")
                return ""

            # Waterfall cascade: Start with the preferred engine, then try others
            if ocr_engine == "paddle": text = try_paddle() or try_tesseract() or try_easyocr()
            elif ocr_engine == "easyocr": text = try_easyocr() or try_paddle() or try_tesseract()
            elif ocr_engine == "tesseract": text = try_tesseract() or try_paddle() or try_easyocr()
            else: text = try_paddle() or try_tesseract() or try_easyocr()
                
            if text:
                # Sanitize the output:
                text = text.replace('I', '!').replace('l', '!').replace('1', '!')
                text = re.sub(r'[^\!\?\.\~\·\s]', '', text).strip()
                
                if text:
                    r.source_text = text
                    r.confidence = 0.5
                    logger.info(f"[Inpainter] Symbol fallback recovered: '{r.source_text}'")
                
        return regions

    # ── Koharu engine methods ─────────────────────────────────────────────────

    def _run_yolo_detect(self, image: Image.Image) -> List[BubbleRegion]:
        """Use YOLO detector from the Koharu pipeline for text region detection."""
        if not self._yolo_model:
            from ultralytics import YOLO
            model_path = self._models_dir / "Detection and Layout" / "comic-text-segmenter.pt"
            logger.info("[Modular] [VRAM] Loading YOLO text detector model...")
            self._yolo_model = YOLO(str(model_path))
            logger.info("[Modular] YOLO text detector loaded.")

        # Run detection with configurable confidence
        conf = float(self.cfg.get("detection_confidence", 0.20))
        results = self._yolo_model(image, verbose=False, conf=conf, iou=0.45)
        regions = []
        for r in results:
            if not r.boxes:
                continue
            
            import cv2
            import numpy as np
            masks = r.masks.xy if r.masks is not None else []
            
            for i, box in enumerate(r.boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                bubble_id = -1
                if masks:
                    for j, mask_pts in enumerate(masks):
                        if len(mask_pts) > 2:
                            mask_cnt = mask_pts.astype(np.float32)
                            dist = cv2.pointPolygonTest(mask_cnt, (cx, cy), measureDist=False)
                            if dist >= 0:
                                bubble_id = j
                                break
                
                # Extract the specific mask points for this bubble if available
                this_mask_pts = None
                if bubble_id != -1 and masks:
                    this_mask_pts = masks[bubble_id]
                
                regions.append(BubbleRegion(
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    source_text="",
                    bubble_id=bubble_id,
                    mask_pts=this_mask_pts
                ))
        return regions

    def _run_ogkalu_combine_detect(self, image: Image.Image) -> List[BubbleRegion]:
        """Use the Ogkalu Combine ONNX detector (detector.onnx)."""
        logger.info("[Modular] Using Ogkalu Combine ONNX detector...")
        # We use the MIT CTD engine but force it to use the ONNX model we just prepared
        return self._run_mit_detect("", image, force_det="ctd")

    def _run_ogkalu_stable_dual_detect(self, image: Image.Image) -> List[BubbleRegion]:
        """
        Ogkalu Stable (YOLOv8 Dual) Engine:
        Uses the standard YOLOv8m text model and YOLOv8m bubble model.
        """
        logger.info("[Modular] Running Ogkalu Stable Dual Engine...")
        from ultralytics import YOLO
        import cv2
        import numpy as np

        # 1. Load models
        text_dir = self._models_dir / "Detection and Layout" / "models" / "ogkalu-text-stable"
        bubble_dir = self._models_dir / "Detection and Layout" / "models" / "ogkalu-bubble-stable"

        text_path = next(text_dir.glob("*.pt"), None)
        bubble_path = next(bubble_dir.glob("*.pt"), None)

        if not text_path or not bubble_path:
            print("\n" + "!"*60)
            print("⚠️  WARNING: Ogkalu Stable Dual models missing! Falling back to MIT Mayo...")
            print(f"Checked: {text_dir} and {bubble_dir}")
            print("Please run: .\\python\\python.exe scripts/download_new_models.py")
            print("!"*60 + "\n")
            return self._run_mit_detect("", image)

        # 2. Run Text Inference
        try:
            logger.info("[Modular] [VRAM] Loading Ogkalu Stable Text model...")
            text_model = YOLO(str(text_path))
            text_res = text_model(image, verbose=False, conf=0.20)
        except Exception as e:
            print("\n" + "!"*60)
            print(f"⚠️  WARNING: [Ogkalu Stable Dual] Text inference failed: {e}")
            print("Falling back to MIT...")
            print("!"*60 + "\n")
            logger.error(f"[Ogkalu Stable Dual] Text inference failed: {e}")
            return self._run_mit_detect("", image)

        # 3. Run Bubble Inference
        try:
            logger.info("[Modular] [VRAM] Loading Ogkalu Stable Bubble model...")
            bubble_model = YOLO(str(bubble_path))
            bubble_res = bubble_model(image, verbose=False, conf=0.25)
        except Exception as e:
            print("\n" + "!"*60)
            print(f"⚠️  WARNING: [Ogkalu Stable Dual] Bubble inference failed: {e}")
            print("Running without bubbles...")
            print("!"*60 + "\n")
            logger.error(f"[Ogkalu Stable Dual] Bubble inference failed: {e}")
            bubble_res = []

        # 4. Logic: Merge Results
        regions = []
        if text_res:
            all_bubbles = []
            if bubble_res:
                for r in bubble_res:
                    if r.masks:
                        all_bubbles.extend(r.masks.xy)
            
            for r in text_res:
                if not r.boxes: continue
                # Match this text to the best bubble shape
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    x1, y1, x2, y2 = coords
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    
                    this_mask_pts = None
                    bubble_id = -1
                    for j, mask_pts in enumerate(all_bubbles):
                        if len(mask_pts) > 2:
                            dist = cv2.pointPolygonTest(mask_pts.astype(np.float32), (cx, cy), False)
                            if dist >= 0:
                                this_mask_pts = mask_pts.astype(int).tolist()
                                bubble_id = j
                                break
                    
                    regions.append(BubbleRegion(
                        x=x1, y=y1, w=x2-x1, h=y2-y1,
                        source_text="",
                        bubble_id=bubble_id,
                        mask_pts=this_mask_pts
                    ))
                            
        return regions

    def _run_manga_ocr_on_regions(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """High-quality Japanese OCR via manga-ocr."""
        from manga_ocr import MangaOCR
        if not hasattr(self, '_mocr') or self._mocr is None:
            model_path = self._models_dir / "OCR" / "manga-ocr-base"
            logger.info("[Modular] [VRAM] Loading MangaOCR model...")
            self._mocr = MangaOCR(str(model_path) if model_path.exists() else None)
        
        for region in regions:
            crop = region.crop(image)
            region.source_text = self._mocr(crop)
            # Manga-OCR doesn't return confidence, use heuristic:
            # - If text is detected: 0.85 confidence (generally reliable)
            # - If empty: 0.0 confidence
            region.confidence = 0.85 if region.source_text and region.source_text.strip() else 0.0
        return regions

    def _run_pororo_ocr(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Elite Ogkalu ONNX Pororo Engine (CRAFT + BrainOCR)."""
        import onnxruntime as ort
        import numpy as np
        import cv2

        try:
            pororo_dir = self._models_dir / "OCR" / "pororo"
            craft_path = pororo_dir / "craft.onnx"
            brain_path = pororo_dir / "brainocr.onnx"

            if not craft_path.exists() or not brain_path.exists():
                raise FileNotFoundError(f"Ogkalu Pororo models missing at: {pororo_dir}")

            # 1. Initialize Sessions with Quiet Logging (No Speed Penalty)
            if not hasattr(self, '_pororo_brain_sess'):
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("[Modular] [VRAM] Loading Pororo OCR models (ONNX)...")
                self._pororo_brain_sess = ort.InferenceSession(str(brain_path), sess_options, providers=providers)
                self._pororo_craft_sess = ort.InferenceSession(str(craft_path), sess_options, providers=providers)

            for region in regions:
                crop = region.crop(image).convert("L")
                w, h = crop.size
                # Elite BrainOCR expects 64px height
                new_w = int(w * (64 / h))
                crop = crop.resize((new_w, 64), Image.LANCZOS)
                
                img_np = np.array(crop).astype(np.float32) / 255.0
                img_np = np.expand_dims(np.expand_dims(img_np, 0), 0)

                inputs = {self._pororo_brain_sess.get_inputs()[0].name: img_np}
                preds = self._pororo_brain_sess.run(None, inputs)[0]
                
                try:
                    # ── GLOBAL COMPATIBILITY SHIELD ──
                    import numpy as np
                    if not hasattr(np, 'bool'): np.bool = bool
                    if not hasattr(np, 'float'): np.float = float
                    if not hasattr(np, 'int'): np.int = int
                    
                    import PIL.Image
                    if not hasattr(PIL.Image, 'ANTIALIAS'):
                        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
                    if not hasattr(PIL.Image, 'Resampling'):
                        class MockResampling: LANCZOS = PIL.Image.LANCZOS
                        PIL.Image.Resampling = MockResampling
                    
                    import torchvision.models.vgg
                    if not hasattr(torchvision.models.vgg, 'model_urls'):
                        torchvision.models.vgg.model_urls = {
                            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
                        }
                    
                    import prrocr
                    if not hasattr(self, '_prrocr_engine'):
                        self._prrocr_engine = prrocr.ocr(lang="ja" if self.cfg.get("source_lang") == "jpn_Jpan" else "ko")
                    region.source_text = " ".join(self._prrocr_engine(np.array(region.crop(image))))
                except Exception as e:
                    logger.warning(f"[Pororo] Decoder failed: {e}")
                    region.source_text = "" 
            return regions
        except Exception as e:
            logger.error(f"[Pororo] Elite Engine failed: {e}")
            raise e

    def _run_manga_ocr_on_regions(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Use Manga-OCR for high-quality Japanese text recognition."""
        if not self._mocr_model:
            from manga_ocr import MangaOcr
            model_path = self._models_dir / "OCR" / "manga-ocr-base"
            self._mocr_model = MangaOcr(str(model_path))
            logger.info("[Modular] MangaOCR loaded.")

        for region in regions:
            try:
                region.source_text = self._mocr_model(region.crop(image))
            except Exception as e:
                logger.warning(f"[MangaOCR] Failed: {e}")
        return regions

    def _run_ppocr_v5_ocr(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Ultra-modern PaddleOCR v4/v5 logic."""
        from paddleocr import PaddleOCR
        if not hasattr(self, '_ppocr_v5') or self._ppocr_v5 is None:
            lang = 'japan' if self.cfg.get("source_lang") == "jpn_Jpan" else 'korean'
            self._ppocr_v5 = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=True)
        
        import numpy as np
        for region in regions:
            crop = np.array(region.crop(image))
            res = self._ppocr_v5.ocr(crop, cls=True)
            if res and res[0]:
                texts = [line[1][0] for line in res[0]]
                region.source_text = " ".join(texts)
        return regions

    def _run_ogkalu_ocr_onnx(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """🏮 Ogaklu OCR (CHN+JPN vertical) ONNX implementation."""
        import onnxruntime as ort
        import numpy as np
        from PIL import Image

        try:
            mocr_dir = self._models_dir / "OCR" / "manga-ocr-onnx"
            encoder_path = mocr_dir / "encoder_model_int8.onnx"
            decoder_path = mocr_dir / "decoder_model_int8.onnx"
            vocab_path = mocr_dir / "vocab.txt"

            if not encoder_path.exists() or not decoder_path.exists() or not vocab_path.exists():
                logger.warning(f"[Ogkalu OCR] Models missing at {mocr_dir}. Falling back to standard MangaOCR.")
                return self._run_manga_ocr_on_regions(image, regions)

            # 1. Initialize Sessions
            if not hasattr(self, '_ogkalu_ocr_encoder') or self._ogkalu_ocr_encoder is None:
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("[Modular] [VRAM] Loading Ogkalu OCR (ONNX)...")
                self._ogkalu_ocr_encoder = ort.InferenceSession(str(encoder_path), sess_options, providers=providers)
                self._ogkalu_ocr_decoder = ort.InferenceSession(str(decoder_path), sess_options, providers=providers)
                
                # Load vocab
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self._ogkalu_ocr_vocab = [line.strip() for line in f.readlines()]

            for region in regions:
                try:
                    # 2. Preprocess
                    # TrOCR typically expects 224x224 RGB
                    crop = region.crop(image).convert("RGB").resize((224, 224), Image.LANCZOS)
                    img_np = np.array(crop).astype(np.float32) / 255.0
                    # Normalization (standard ViT)
                    img_np = (img_np - 0.5) / 0.5
                    img_np = img_np.transpose(2, 0, 1) # HWC -> CHW
                    img_np = np.expand_dims(img_np, 0) # CHW -> BCHW

                    # 3. Run Encoder
                    encoder_inputs = {self._ogkalu_ocr_encoder.get_inputs()[0].name: img_np}
                    encoder_outputs = self._ogkalu_ocr_encoder.run(None, encoder_inputs)
                    last_hidden_state = encoder_outputs[0]

                    # 4. Greedy Decode
                    # Standard BERT/ViT-GPT2: BOS=2 [CLS], EOS=3 [SEP], PAD=0 [PAD]
                    input_ids = np.array([[2]], dtype=np.int64)
                    generated_tokens = []
                    
                    for _ in range(128): # Max tokens per bubble
                        decoder_inputs = {
                            self._ogkalu_ocr_decoder.get_inputs()[0].name: input_ids,
                            self._ogkalu_ocr_decoder.get_inputs()[1].name: last_hidden_state
                        }
                        decoder_outputs = self._ogkalu_ocr_decoder.run(None, decoder_inputs)
                        logits = decoder_outputs[0]
                        
                        # Get next token (greedy)
                        next_token = np.argmax(logits[0, -1, :])
                        if next_token == 3: # SEP/EOS
                            break
                        generated_tokens.append(int(next_token))
                        
                        # Append and continue
                        input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=1)

                    # 5. Convert tokens to text
                    text = ""
                    for token in generated_tokens:
                        if 0 <= token < len(self._ogkalu_ocr_vocab):
                            t = self._ogkalu_ocr_vocab[token]
                            # Handle common subword markers if present, though likely char-level
                            if t.startswith("##"):
                                text += t[2:]
                            elif t not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
                                text += t
                    region.source_text = text.strip()
                    region.confidence = 0.9
                except Exception as e:
                    logger.warning(f"[Ogkalu OCR] Region processing failed: {e}")
                    region.source_text = ""
            return regions
        except Exception as e:
            logger.error(f"[Ogkalu OCR] Engine runtime failed: {e}")
            return self._run_manga_ocr_on_regions(image, regions)

    def _run_mayo_github_detect(self, image: Image.Image) -> List[BubbleRegion]:
        """
        Mayo Github (Classic) Engine:
        Uses Mayocream's classic comic-text-detector + speech-bubble-segmentation.
        """
        logger.info("[Modular] Running Mayo Github (Classic) Engine...")
        from ultralytics import YOLO
        import cv2
        import numpy as np

        # 1. Load models
        text_dir = self._models_dir / "Detection and Layout" / "models" / "mayo-text-classic"
        bubble_dir = self._models_dir / "Detection and Layout" / "models" / "mayo-bubble-seg"

        text_path = next(text_dir.glob("*.pt"), None) or next(text_dir.glob("*.safetensors"), None)
        bubble_path = next(bubble_dir.glob("*.pt"), None) or next(bubble_dir.glob("*.safetensors"), None)

        if not text_path or not bubble_path:
            print("\n" + "!"*60)
            print("⚠️  WARNING: Mayo Github models missing! Falling back to MIT Mayo...")
            print(f"Checked: {text_dir} and {bubble_dir}")
            print("Please run: .\\python\\python.exe scripts/download_new_models.py")
            print("!"*60 + "\n")
            return self._run_mit_detect("", image)

        # 2. Run Mayo Github (Classic Optimized Engine)
        try:
            # This uses the high-performance MIT CTD engine which is the "Classic" Mayo logic
            return self._run_mit_detect("", image, force_det="ctd")
        except Exception as e:
            logger.error(f"[Mayo Github] Classic optimized engine failed: {e}")
            return self._run_mit_detect("", image)

        # 4. Logic: Merge Results
        regions = []
        for box in text_boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            parent = None
            for b_mask in bubbles:
                if cv2.pointPolygonTest(b_mask.astype(np.float32), (cx, cy), False) >= 0:
                    parent = b_mask.astype(int).tolist()
                    break
            regions.append(BubbleRegion(x=x1, y=y1, w=x2-x1, h=y2-y1, source_text="", mask_pts=parent if parent is not None else []))
                             
        return regions

    def _run_mit_ocr_on_regions(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Use MIT Mayo (JPN) 48px OCR for fast Japanese recognition."""
        from manga_translator.ocr import OCR
        if not hasattr(self, '_mit_ocr') or self._mit_ocr is None:
            # We use the standard MIT OCR engine (Model48pxOCR)
            self._mit_ocr = OCR('48px', device=self.device)
            logger.info("[Modular] MIT Mayo (JPN) OCR loaded.")
        
        import numpy as np
        for region in regions:
            try:
                crop = np.array(region.crop(image))
                # MIT OCR expects a list of crops and returns a list of results
                res = self._mit_ocr.recognize([crop])
                if res and len(res) > 0:
                    region.source_text = res[0].text
            except Exception as e:
                logger.warning(f"[MIT OCR] Failed on region: {e}")
        return regions

    # ── Font & Inpaint Logic ──────────────────────────────────────────────────

    def _run_yuzumarker_font_detection(self, image: Image.Image, region: BubbleRegion) -> str:
        """Use Yuzumarker ONNX model to detect font style from a crop."""
        if not hasattr(self, '_font_det_session') or self._font_det_session is None:
            try:
                import onnxruntime as ort
                model_path = self._models_dir / "OCR" / "font-detection" / "font-detector_int8.onnx"
                if not model_path.exists():
                    model_path = self._models_dir / "OCR" / "font-detection" / "font-detector.onnx"
                if not model_path.exists():
                    print("\n" + "!"*60)
                    print("⚠️  WARNING: Yuzumarker font model missing! Falling back...")
                    print("Please run: .\\python\\python.exe scripts/download_new_models.py")
                    print("!"*60 + "\n")
                    return "default"
                self._font_det_session = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                logger.info("[Modular] Yuzumarker font detection loaded.")
            except Exception as e:
                logger.error("[Modular] Failed to load Yuzumarker font detection: %s", e)
                return "default"

        try:
            import numpy as np
            import cv2
            # Model expects: [1, 3, 512, 512] — RGB, 512x512
            crop = np.array(region.crop(image).convert("RGB"))
            crop = cv2.resize(crop, (512, 512))
            crop = crop.astype(np.float32) / 255.0
            # Normalize with ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop = (crop - mean) / std
            # HWC -> CHW -> NCHW
            crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, 0).astype(np.float32)
            
            inputs = {self._font_det_session.get_inputs()[0].name: crop}
            outputs = self._font_det_session.run(None, inputs)
            idx = int(np.argmax(outputs[0]))
            # Yuzumarker classes: 0=normal, 1=bold, 2=italic, 3=handwritten, 4=sfx
            style_map = {0: "normal", 1: "bold", 2: "italic", 3: "handwritten", 4: "sfx"}
            return style_map.get(idx, "default")
        except Exception as e:
            logger.warning("[Yuzumarker] Font detection failed: %s", e)
            return "default"

    def _apply_sfx_strictness_filter(self, image: Image.Image, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """
        Filters out regions that are likely artistic noise/SFX based on 
        the variance of the background pixels and OCR confidence.
        """
        if not self.cfg.get("enable_sfx_filter", True):
            return regions
            
        strictness = float(self.cfg.get("sfx_strictness", 1.0))
        if strictness <= 0:
            return regions

        filtered = []
        for r in regions:
            # Inside a detected bubble -> Keep it immediately to prevent dropping dialogue
            if r.bubble_id != -1:
                filtered.append(r)
                continue

            # Free-floating text (or MIT detector which sets bubble_id=-1 for everything)
            crop_pil = r.crop(image)
            crop_np = np.array(crop_pil.convert("L"))
            var = np.var(crop_np)
            
            score = r.confidence * 100 / (var + 1)
            threshold = 0.5 * strictness
            text_len = len(r.source_text.strip()) if r.source_text else 0
            
            # If the text is a phrase (>= 3 chars), it's usually dialogue.
            # We apply a moderate threshold to protect it, but it MUST pass to prevent hallucinated SFX.
            if text_len >= 3:
                if score >= threshold:
                    filtered.append(r)
                else:
                    logger.info("[SFX Filter] Dropping hallucinated long text: '%s', score=%.2f", r.source_text, score)
                continue

            # Short text (1-2 chars). This is where most SFX noise lives.
            # We strictly require a high score (clean background) to keep it.
            if score >= (threshold * 2.0):
                filtered.append(r)
            else:
                logger.info("[SFX Filter] Dropping SFX/Noise: text='%s', score=%.2f, var=%.1f", 
                            r.source_text, score, var)
        
        return filtered

    def _apply_nuisance_filter(self, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """
        Filters out 'nuisance' detections: free-floating, very small area, and tiny text length (1-2 chars).
        Real text (even short) should be kept if it's large or inside a bubble.
        """
        if not self.cfg.get("enable_nuisance_filter", False):
            return regions
            
        filtered = []
        for r in regions:
            # Must be free-floating (no bubble parent detected)
            is_free_floating = (r.bubble_id == -1)
            
            # Check size constraints
            area = r.w * r.h
            # Tiny area (e.g. 50x50 = 2500 pixels max)
            is_tiny_area = area < 2500
            
            text_len = len(r.source_text.strip()) if r.source_text else 0
            
            # Common junk characters often misidentified as text by OCR when scanning speed lines/screentones
            junk_chars = set("「」わぁあ、。.,!?~ー|/\\lI1")
            
            is_junk = (text_len <= 2 and all(c in junk_chars for c in r.source_text.strip() if c.strip()))
            
            # If it's free-floating AND tiny AND (has no text OR is ONLY junk characters)
            if is_free_floating and is_tiny_area and (text_len == 0 or is_junk):
                logger.info(f"[Nuisance Filter] Dropping nuisance: text='{r.source_text}', size={r.w}x{r.h}, free-floating=True")
                continue
                
            filtered.append(r)
            
        return filtered



    def unload_models(self):
        """Free VRAM by unloading lazy-loaded detection/OCR models."""
        if self._yolo_model:
            logger.info("[Modular] [VRAM] Unloading YOLO model from VRAM...")
            del self._yolo_model
            self._yolo_model = None
        if self._mocr_model:
            logger.info("[Modular] [VRAM] Unloading MangaOCR model from VRAM...")
            del self._mocr_model
            self._mocr_model = None
            
        if hasattr(self, '_mocr') and self._mocr is not None:
            logger.info("[Modular] [VRAM] Unloading MangaOCR (MangaOCR object) from VRAM...")
            del self._mocr
            self._mocr = None
            
        if hasattr(self, '_pororo_brain_sess'):
            logger.info("[Modular] [VRAM] Unloading Pororo OCR sessions from VRAM...")
            del self._pororo_brain_sess
            if hasattr(self, '_pororo_craft_sess'):
                del self._pororo_craft_sess
                
        if hasattr(self, '_ogkalu_ocr_encoder') and self._ogkalu_ocr_encoder is not None:
            logger.info("[Modular] [VRAM] Unloading Ogkalu OCR sessions from VRAM...")
            del self._ogkalu_ocr_encoder
            del self._ogkalu_ocr_decoder
            self._ogkalu_ocr_encoder = None
            self._ogkalu_ocr_decoder = None
        
        if hasattr(self, '_ogkalu_inpaint_sess'):
            logger.info("[Modular] [VRAM] Unloading Ogkalu Inpaint session from VRAM...")
            del self._ogkalu_inpaint_sess
            self._ogkalu_inpaint_sess = None
            
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[Modular] [VRAM] VRAM Cache cleared.")
        except ImportError:
            pass

    def inpaint(self, image: Image.Image, regions: List[BubbleRegion]) -> Image.Image:
        """
        Remove the original text from bubble regions (clean background).
        Returns a new PIL Image with text erased.
        """
        engine = self.cfg.get("inpaint_engine", "lama").lower()
        use_seg = "_segmented" in engine
        base_engine = engine.replace("_segmented", "")
        
        if base_engine == "aot":
            return self._run_aot_inpaint(image, regions, use_segmentation=use_seg)
        elif base_engine == "ogkalu":
            return self._run_ogkalu_inpaint(image, regions, use_segmentation=use_seg)
        elif base_engine == "panelcleaner":
            return self._run_panelcleaner_inpaint(image, regions, use_segmentation=use_seg)
        elif base_engine == "solid":
            global_mask = None
            if use_seg:
                global_mask = self._generate_precise_text_mask(image, regions)
                
            result = image.copy()
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_seg, global_mask=global_mask)
            return result
            
        return self._run_mit_inpaint(image, regions)

    def _generate_precise_text_mask(self, image: Image.Image, regions: List[BubbleRegion]) -> "np.ndarray":
        """
        Generate a precise context-aware text mask using ComicTextDetector.
        Intersects the global CTD mask with the region bounding boxes.
        """
        import numpy as np
        import cv2
        from core.panelcleaner_wrapper import PanelCleanerPipeline
        
        np_img = np.array(image)
        pipeline = PanelCleanerPipeline(device=self.device)
        ctd_mask = pipeline.detect_text_mask(np_img)
        
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for r in regions:
            # Expand bounding box slightly
            pad = 6
            x1 = max(0, r.x - pad)
            y1 = max(0, r.y - pad)
            x2 = min(image.width, r.x + r.w + pad)
            y2 = min(image.height, r.y + r.h + pad)
            
            # Keep only the CTD mask within this text region
            region_mask = ctd_mask[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], region_mask)
            
        return mask

    def _run_panelcleaner_inpaint(self, image: Image.Image, regions: List[BubbleRegion], use_segmentation: bool = False) -> Image.Image:
        """
        Use PanelCleaner's LaMa model for inpainting.
        """
        try:
            import numpy as np
            import cv2
            from core.panelcleaner_wrapper import PanelCleanerPipeline
            
            if use_segmentation:
                mask = self._generate_precise_text_mask(image, regions)
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                for r in regions:
                    mask[r.y:r.y+r.h, r.x:r.x+r.w] = 255

            if np.max(mask) == 0:
                return image

            np_img = np.array(image)
            pipeline = PanelCleanerPipeline(device=self.device)
            inpainted = pipeline.inpaint_lama(np_img, mask)
            
            logger.info("[PanelCleaner] Inpainting complete.")
            return Image.fromarray(inpainted)
        except Exception as exc:
            logger.warning("[PanelCleaner] Inpainting failed (%s). Falling back to Solid Fill.", exc)
            result = image.copy()
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_segmentation)
            return result

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

        # ── Font Style Detection (Yuzumarker) ─────────────────────────────
        font_det_engine = self.cfg.get("font_detection_engine", "default")
        if font_det_engine == "yuzumarker":
            for region in regions:
                if region.translated_text:
                    detected_style = self._run_yuzumarker_font_detection(image, region)
                    region.font_style = detected_style
                    logger.debug("[Yuzumarker] Region style: %s", detected_style)

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
            bg_color = self._detect_bubble_bg_color(result, region)
            
            if color == "auto":
                is_dark = self._is_dark_background(bg_color)
                color = "#FFFFFF" if is_dark else "#000000"
                
            # If in a bubble, outline matches the bubble's background. Otherwise contrasting.
            if region.bubble_id != -1:
                bg_hex = f"#{bg_color[0]:02X}{bg_color[1]:02X}{bg_color[2]:02X}"
                outline_color = bg_hex
            else:
                outline_color = "#000000" if color.upper() in ("#FFFFFF", "#FFF", "WHITE") else "#FFFFFF"

            font = self._load_font(font_path, font_size)
            self._draw_text_in_bubble(draw, region, region.translated_text, font, color, outline_color)

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
                    bubble_id=region.bubble_id
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
            
            # GATED MERGING LOGIC: Only merge if bubble_id matches
            if current.bubble_id == next_r.bubble_id:
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
                    translated_text=combined_trans,
                    confidence=min(current.confidence, next_r.confidence),
                    font_cfg=current.font_cfg,
                    bubble_id=current.bubble_id
                )
            else:
                merged.append(current)
                current = next_r

        merged.append(current)
        return self._deduplicate_contained_regions(merged)

    def _deduplicate_contained_regions(self, regions: List[BubbleRegion]) -> List[BubbleRegion]:
        """Remove smaller regions that are contained within larger, better-defined parent regions."""
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
                
                # If >80% of small bubble is inside large bubble, the small bubble 
                # is likely a redundant fragment. Remove it to keep the whole text block.
                if inter_area > 0.8 * small_area:
                    remove_indices.add(i)
            
        keep = [r for idx, r in enumerate(sorted_regions) if idx not in remove_indices]
        # Return in original top-to-bottom order
        return sorted(keep, key=lambda r: (r.y, r.x))

    def _run_aot_inpaint(self, image: Image.Image, regions: List[BubbleRegion], use_segmentation: bool = False) -> Image.Image:
        """
        Use AOT-GAN ONNX model from Pipeline Koharu for high-detail inpainting.
        Falls back to MIT inpainting if the model is not found or fails.
        """
        aot_model_path = self._models_dir / "Inpainting" / "aot-inpainting" / "aot.onnx"
        if not aot_model_path.exists():
            logger.warning("[AOT] Model not found at %s. Falling back to Solid Fill.", aot_model_path)
            result = image.copy()
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_segmentation)
            return result

        try:
            import onnxruntime as ort
            import numpy as np

            # Create a full-page mask
            if use_segmentation:
                mask = self._generate_precise_text_mask(image, regions)
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                for r in regions:
                    mask[r.y:r.y+r.h, r.x:r.x+r.w] = 255

            # If nothing to inpaint, return image
            if np.max(mask) == 0:
                return image

            # 1.5 Resolution Safety Cap
            MAX_DIM = 2048
            orig_h, orig_w = image.height, image.width
            if max(orig_h, orig_w) > MAX_DIM:
                scale = MAX_DIM / max(orig_h, orig_w)
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                img_to_proc = np.array(image.resize((new_w, new_h), Image.LANCZOS))
                import cv2
                mask_to_proc = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                h, w = new_h, new_w
                logger.info("[AOT] Page too large (%dx%d), scaling to %dx%d for VRAM safety.", orig_w, orig_h, new_w, new_h)
            else:
                img_to_proc = np.array(image)
                mask_to_proc = mask
                h, w = orig_h, orig_w

            # Convert to numpy arrays
            mask_3ch_full = np.stack([mask]*3, axis=-1) / 255.0

            # Preprocess image
            np_img = img_to_proc.astype(np.float32) / 255.0
            
            h, w = np_img.shape[:2]
            
            # AOT-GAN requires dimensions to be multiples of 8
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                np_img = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                mask_padded = np.pad(mask_to_proc, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            else:
                mask_padded = mask_to_proc

            # AOT expects [1, 3, H, W] image and [1, 1, H, W] mask
            # Image is scaled to [-1, 1], Mask is scaled to [0, 1]
            img_tensor  = (np_img * 2.0 - 1.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            mask_tensor = (mask_padded / 255.0)[np.newaxis, np.newaxis].astype(np.float32)

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            sess = ort.InferenceSession(str(aot_model_path), sess_options=sess_options, providers=providers)

            inp_name  = sess.get_inputs()[0].name
            mask_name = sess.get_inputs()[1].name
            out_name  = sess.get_outputs()[0].name

            # Run inference
            img_tensor = img_tensor * (1.0 - mask_tensor)
            out_tensor = sess.run([out_name], {inp_name: img_tensor, mask_name: mask_tensor})[0]

            # Crop padding back off
            if pad_h > 0 or pad_w > 0:
                out_tensor = out_tensor[:, :, :h, :w]

            # Post-process [-1, 1] back to [0, 255]
            out = (out_tensor[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0
            out = np.clip(out, 0, 255)

            # Scale back up if necessary
            if out.shape[0] != image.height or out.shape[1] != image.width:
                import cv2
                out = cv2.resize(out, (image.width, image.height), interpolation=cv2.INTER_LANCZOS4)

            original_full = np.array(image)
            composite = (original_full * (1 - mask_3ch_full) + out * mask_3ch_full).astype(np.uint8)

            logger.info("[AOT] Inpainting complete via AOT-GAN ONNX.")
            return Image.fromarray(composite)

        except Exception as exc:
            logger.warning("[AOT] Inpainting failed (%s). Falling back to Solid Fill.", exc)
            result = image.copy()
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_segmentation)
            return result

    def _run_ogkalu_inpaint(self, image: Image.Image, regions: List[BubbleRegion], use_segmentation: bool = False) -> Image.Image:
        """
        Elite Ogkalu LaMa Inpainting Engine.
        Uses the dynamic-size LaMa ONNX model optimized for manga restoration.
        """
        model_path = self._models_dir / "Inpainting" / "ogkalu" / "lama-manga-dynamic.onnx"
        
        if not model_path.exists():
            logger.warning("[Ogkalu Inpaint] Model not found at %s. Falling back to Solid Fill.", model_path)
            print("\n" + "!"*60)
            print("⚠️  WARNING: Ogkalu LaMa model missing!")
            print(f"Please download lama-manga-dynamic.onnx and place it in: {model_path.parent}")
            print("!"*60 + "\n")
            result = image.copy()
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_segmentation)
            return result

        try:
            import onnxruntime as ort
            import numpy as np
            import cv2

            # 1. Create Mask
            if use_segmentation:
                mask = self._generate_precise_text_mask(image, regions)
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                for r in regions:
                    # Expand mask slightly for cleaner edges in LaMa
                    kernel = np.ones((5,5), np.uint8)
                    region_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                    region_mask[r.y:r.y+r.h, r.x:r.x+r.w] = 255
                    dilated = cv2.dilate(region_mask, kernel, iterations=1)
                    mask = cv2.bitwise_or(mask, dilated)

            if np.max(mask) == 0:
                return image

            # 2. Preprocess with Resolution Safety Cap
            # We scale down to 2048px if necessary to avoid VRAM OOM.
            MAX_DIM = 2048
            orig_h, orig_w = image.height, image.width
            
            if max(orig_h, orig_w) > MAX_DIM:
                scale = MAX_DIM / max(orig_h, orig_w)
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                # Resize image and mask
                img_np_orig = np.array(image.resize((new_w, new_h), Image.LANCZOS))
                mask_np_orig = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                h, w = new_h, new_w
                logger.info("[Ogkalu Inpaint] Page too large (%dx%d), scaling to %dx%d for VRAM safety.", orig_w, orig_h, new_w, new_h)
            else:
                img_np_orig = np.array(image)
                mask_np_orig = mask
                h, w = orig_h, orig_w
            
            # Pad to multiple of 8 (standard for LaMa)
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            
            img_np = img_np_orig.astype(np.float32) / 255.0
            mask_np = (mask_np_orig.astype(np.float32) / 255.0)
            
            if pad_h > 0 or pad_w > 0:
                img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            # [1, 3, H, W] and [1, 1, H, W]
            img_tensor = img_np.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            mask_tensor = mask_np[np.newaxis, np.newaxis].astype(np.float32)

            # 3. Inference
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if not hasattr(self, '_ogkalu_inpaint_sess') or self._ogkalu_inpaint_sess is None:
                logger.info("[Ogkalu Inpaint] [VRAM] Loading Dynamic LaMa Engine...")
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3
                self._ogkalu_inpaint_sess = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)

            inputs = {
                self._ogkalu_inpaint_sess.get_inputs()[0].name: img_tensor,
                self._ogkalu_inpaint_sess.get_inputs()[1].name: mask_tensor
            }
            out_tensor = self._ogkalu_inpaint_sess.run(None, inputs)[0]

            # 4. Post-process
            if pad_h > 0 or pad_w > 0:
                out_tensor = out_tensor[:, :, :h, :w]
            
            out_img = out_tensor[0].transpose(1, 2, 0)
            out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)

            # If we scaled down, scale the result back up to match original image
            if out_img.shape[0] != image.height or out_img.shape[1] != image.width:
                out_img = cv2.resize(out_img, (image.width, image.height), interpolation=cv2.INTER_LANCZOS4)

            # Composite (only replace masked areas)
            original_full = np.array(image)
            mask_3ch = np.stack([mask/255.0]*3, axis=-1)
            final_np = (original_full * (1 - mask_3ch) + out_img * mask_3ch).astype(np.uint8)

            logger.info("[Ogkalu Inpaint] Inpainting complete.")
            return Image.fromarray(final_np)

        except Exception as exc:
            logger.error("[Ogkalu Inpaint] Failed: %s", exc)
            result = image.copy()
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_segmentation)
            return result

    # ── MIT integration (detection + inpainting) ──────────────────────────────

    def _run_mit_detect(self, image_path: str, image: Image.Image, force_det=None) -> List[BubbleRegion]:
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
                "use_gpu": self.device.startswith("cuda"),
                "verbose": False,
                "kernel_size": 3,
                "pre_dict": None,
                "post_dict": None,
            }
            translator_obj = MangaTranslator(params)

            # Build config — only detect + OCR, skip translation/rendering
            cfg = Config()
            det_key = force_det if force_det else self._mit_cfg.get("detector", "default")
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
                    device=self.device, verbose=False,
                )

            cleaned = asyncio.run(_inpaint())
            return Image.fromarray(cleaned)

        except Exception as exc:
            logger.warning("MIT inpaint failed (%s), using smart-fill fallback.", exc)
            result = image.copy()
            use_segmentation = self.cfg.get("use_segmentation", False)
            for r in regions:
                self._clean_region(result, r, use_segmentation=use_segmentation)
            return result

    def _clean_region(self, image: Image.Image, region: "BubbleRegion", use_segmentation: bool = False, global_mask = None):
        """
        Inpaint a single region with a solid background color.
        Uses a feathered mask for smoother blending.
        """
        img_w, img_h = image.size
        x1, y1, x2, y2 = region.bbox

        # Expand region slightly to ensure we cover the text fully
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

        # Create mask for blending
        if use_segmentation and global_mask is not None:
            import numpy as np
            patch_mask_np = global_mask[fy1:fy2, fx1:fx2]
            mask = Image.fromarray(patch_mask_np).convert("L")
            # Slightly soften the precise text mask for blending
            mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))
        else:
            # Create a soft-edged rectangular mask for blending (feathered edges)
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
        outline_color: str = None
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
        
        if outline_color is None:
            outline_color = "#000000" if color.upper() in ("#FFFFFF", "#FFF", "WHITE") else "#FFFFFF"

        for i, line in enumerate(lines):
            y = start_y + i * line_h
            line_w = int(self._get_mixed_textlength(line, font, emoji_font, draw))
            
            x = region.x + (region.w - line_w) // 2
            x = max(bubble_left, min(x, bubble_right - line_w))

            self._draw_mixed_line(draw, x, y, line, font, emoji_font, color, outline_color)
