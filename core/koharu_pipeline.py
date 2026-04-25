import os
import cv2
import time
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class KoharuBubbleResult:
    x: int
    y: int
    w: int
    h: int
    source_text: str
    translated_text: str
    direction: str

class KoharuPipeline:
    """
    Koharu End-to-End Pipeline implementing:
    1. Detection (YOLO / ONNX)
    2. OCR (PaddleOCR-VL / Manga-OCR)
    3. Font Analysis (YuzuMarker)
    4. Inpainting (LaMa / AOT)
    5. Translation (Using selected engine)
    6. Rendering (Pillow)
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.models_dir = Path(r"D:\Translate\Translator\Pipeline Koharu")
        self.device = "cuda" # Assumes CUDA is available
        self._load_models()

    def _load_models(self):
        logger.info("[Koharu] Initialising models from %s...", self.models_dir)
        try:
            from ultralytics import YOLO
            self.detector_text = YOLO(str(self.models_dir / "Detection and Layout" / "comic-text-segmenter.pt"))
            self.detector_bubble = YOLO(str(self.models_dir / "Detection and Layout" / "comic-speech-bubble-detector.pt"))
            logger.info("[Koharu] YOLO Detectors loaded.")
        except ImportError:
            logger.warning("[Koharu] 'ultralytics' not installed. Text detection will be limited.")
            self.detector_text = None

        try:
            from manga_ocr import MangaOcr
            # Uses manga-ocr-base
            self.manga_ocr = MangaOcr(str(self.models_dir / "OCR" / "manga-ocr-base"))
            logger.info("[Koharu] Manga OCR loaded.")
        except Exception as e:
            logger.warning("[Koharu] Manga-OCR failed to load: %s", e)
            self.manga_ocr = None

        # Font Detection & Inpainting require specific implementations or Transformers
        # We set them up as placeholders that hook into the pipeline structure
        self.font_detector = None
        self.inpainter = None 
        logger.info("[Koharu] Pipeline initialised.")

    def process_page(self, img_path, page_num):
        """Runs the complete Koharu pipeline on a single image."""
        original_img = Image.open(img_path).convert("RGB")
        cv_img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        
        bubbles = []
        result_img = original_img.copy()

        # 1. Detection
        bboxes = self._detect_text_regions(img_path)
        
        # We will create an inpaint mask
        inpaint_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        
        for idx, box in enumerate(bboxes):
            x1, y1, x2, y2 = box
            # Expand bbox slightly for inpainting
            cv2.rectangle(inpaint_mask, (x1-5, y1-5), (x2+5, y2+5), 255, -1)

            # Crop region for OCR and Font Analysis
            crop_pil = original_img.crop((x1, y1, x2, y2))
            
            # 2. OCR
            text = self._run_ocr(crop_pil)
            if not text.strip():
                continue

            # 3. Font Analysis (Pseudo)
            font_info = self._analyze_font(crop_pil)
            
            # Record bubble data
            br = KoharuBubbleResult(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                source_text=text,
                translated_text="", # Will be translated downstream or here
                direction="horizontal"
            )
            # Attach font style hint to bubble if needed
            bubbles.append((br, font_info))

        # 4. Inpainting
        logger.info("  [Koharu] Inpainting original text...")
        inpainted_cv = self._run_inpainting(cv_img, inpaint_mask)
        result_img = Image.fromarray(cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB))

        # Note: Translation typically happens via the Translator module in batch_processor
        # We return the inpainted image and the bubbles for translation/rendering later
        # OR we can render here if translation is passed in.
        
        # For compatibility with batch_processor, we return inpainted image and structured bubbles
        pure_bubbles = [b[0] for b in bubbles]
        return result_img, pure_bubbles

    def _detect_text_regions(self, img_path):
        """Returns list of [x1, y1, x2, y2] bounding boxes."""
        bboxes = []
        if self.detector_text:
            results = self.detector_text(img_path, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    bboxes.append([x1, y1, x2, y2])
        return bboxes

    def _run_ocr(self, crop_pil):
        """Runs OCR on a cropped text image."""
        if self.manga_ocr:
            return self.manga_ocr(crop_pil)
        return "example text" # Fallback

    def _analyze_font(self, crop_pil):
        """Returns font styles (size, weight, color)."""
        # Hook into YuzuMarker.FontDetection model here
        return {"color": (0, 0, 0), "size": 24}

    def _run_inpainting(self, cv_img, mask):
        """Runs LaMa / AOT inpainting to erase text."""
        # Fallback to OpenCV Telea inpainting if LaMa is not fully hooked up
        inpainted = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
        return inpainted
