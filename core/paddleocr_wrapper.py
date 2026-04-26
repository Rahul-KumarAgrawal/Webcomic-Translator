import os
import sys
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ── Redirect ALL Paddle downloads to D drive ─────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PADDLE_CACHE = os.path.join(_ROOT, "model", "paddle_cache")
os.makedirs(_PADDLE_CACHE, exist_ok=True)

# Environment variables for various Paddle components
os.environ["PPOCR_HOME"]    = _PADDLE_CACHE
os.environ["PADDLE_HOME"]   = _PADDLE_CACHE
os.environ["HUB_HOME"]      = _PADDLE_CACHE
os.environ["FLAGS_model_dir"] = _PADDLE_CACHE

# ── DLL Fix: Add NVIDIA bin folders to PATH ──────────────────────────────────
import site
for s_path in site.getsitepackages():
    nvidia_dir = os.path.join(s_path, "nvidia")
    if os.path.exists(nvidia_dir):
        for sub in os.listdir(nvidia_dir):
            bin_path = os.path.join(nvidia_dir, sub, "bin")
            if os.path.exists(bin_path):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_path)
                    except Exception:
                        pass
                # Legacy PATH addition for older Python/Paddle versions
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

# ── Load PaddleOCR with Monkey-patching ──────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    # Monkey-patch BASE_DIR so models download to D: not C:\Users\...\.paddleocr
    # This is required for PaddleOCR 2.x
    import paddleocr.paddleocr as _ppocr_mod
    _ppocr_mod.BASE_DIR = _PADDLE_CACHE
except ImportError:
    logger.warning("paddleocr pip package not found. Attempting to load from D:\\Translate\\PaddleOCR-main")
    sys.path.insert(0, r"D:\Translate\PaddleOCR-main")
    try:
        from paddleocr import PaddleOCR
        import paddleocr.paddleocr as _ppocr_mod
        _ppocr_mod.BASE_DIR = _PADDLE_CACHE
    except ImportError as e:
        logger.error(f"Failed to load PaddleOCR natively or from source: {e}")
        PaddleOCR = None

_paddle_ocr_instance = None
_paddle_last_lang = None

def get_paddle_ocr(lang="japan"):
    """
    Lazy load PaddleOCR 2.8.1. Re-initializes if the target language changes.
    """
    global _paddle_ocr_instance, _paddle_last_lang
    if PaddleOCR is None:
        raise ImportError("PaddleOCR could not be loaded.")
        
    if _paddle_ocr_instance is None or _paddle_last_lang != lang:
        logger.info(f"Initializing PaddleOCR (Stable 2.8.1) for language: {lang}")
        # Stable 2.x arguments
        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=True, 
            lang=lang, 
            show_log=False, 
            use_gpu=True, 
            gpu_mem=500
        )
        _paddle_last_lang = lang
        
    return _paddle_ocr_instance

def map_lang_to_paddle(lang_code: str) -> str:
    """Maps ISO language codes to PaddleOCR codes."""
    lang_code = lang_code.lower()
    if "jp" in lang_code or "japan" in lang_code: return "japan"
    if "ko" in lang_code or "kor" in lang_code: return "korean"
    if "zh" in lang_code or "ch" in lang_code:
        if "hant" in lang_code or "trad" in lang_code: return "chinese_cht"
        return "ch"
    if "en" in lang_code or "eng" in lang_code: return "en"
    if "fr" in lang_code or "fra" in lang_code: return "fr"
    if "es" in lang_code or "spa" in lang_code: return "es"
    if "de" in lang_code or "ger" in lang_code: return "german"
    if "it" in lang_code or "ita" in lang_code: return "it"
    if "pt" in lang_code or "por" in lang_code: return "pt"
    if "ru" in lang_code or "rus" in lang_code: return "ru"
    if "ar" in lang_code or "ara" in lang_code: return "ar"
    if "hi" in lang_code or "hin" in lang_code: return "hi"
    return "japan"

def run_paddle_ocr_on_regions(image: Image.Image, regions: list, cfg: dict) -> list:
    """Runs OCR on cropped regions using PaddleOCR 2.8.1."""
    if not regions:
        return regions
        
    source_lang_hint = cfg.get("source_lang_override") or "japan"
    paddle_lang = map_lang_to_paddle(source_lang_hint)
    
    ocr = get_paddle_ocr(paddle_lang)
    img_np = np.array(image.convert("RGB"))
    img_np = img_np[:, :, ::-1] # RGB to BGR for OpenCV-based OCR
    
    for region in regions:
        x1, y1, x2, y2 = region.bbox
        pad = 5
        h, w = img_np.shape[:2]
        crop_y1, crop_y2 = max(0, int(y1)-pad), min(h, int(y2)+pad)
        crop_x1, crop_x2 = max(0, int(x1)-pad), min(w, int(x2)+pad)
        
        crop = img_np[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
            
        # Run PaddleOCR (Stable 2.x format)
        result = ocr.ocr(crop, cls=True)
        
        if result and result[0]:
            line_entries = []
            for line in result[0]:
                if len(line) >= 2 and len(line[1]) >= 1:
                    text = line[1][0]
                    box = line[0]
                    x_center = sum(pt[0] for pt in box) / 4
                    y_center = sum(pt[1] for pt in box) / 4
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    box_w = max(xs) - min(xs)
                    box_h = max(ys) - min(ys)
                    line_entries.append((x_center, y_center, text, box_w, box_h))
            
            if line_entries:
                is_cjk = paddle_lang in ("japan", "korean", "ch", "chinese_cht")
                avg_aspect = sum(e[4] / max(e[3], 1) for e in line_entries) / len(line_entries)
                is_vertical = is_cjk and avg_aspect > 1.5

                if is_cjk and is_vertical:
                    line_entries.sort(key=lambda e: (-round(e[0] / 15), e[1]))
                else:
                    line_entries.sort(key=lambda e: (round(e[1] / 15), e[0]))
                
                lines = [entry[2] for entry in line_entries]
                new_text = "\n".join(lines)
                region.source_text = new_text
                
    return regions
