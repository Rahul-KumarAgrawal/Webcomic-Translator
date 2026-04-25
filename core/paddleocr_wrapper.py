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

os.environ["PPOCR_HOME"]    = _PADDLE_CACHE           # PaddleOCR model cache
os.environ["PADDLE_HOME"]   = _PADDLE_CACHE           # PaddlePaddle general cache
os.environ["PADDLEX_HOME"]  = _PADDLE_CACHE           # PaddleX model cache
os.environ["HUB_HOME"]      = _PADDLE_CACHE           # PaddleHub cache
os.environ["FLAGS_model_dir"] = _PADDLE_CACHE          # Paddle inference cache
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # skip slow connectivity check

# Fallback: if paddleocr isn't installed via pip, try loading from the user's downloaded repo
try:
    from paddleocr import PaddleOCR
    # Monkey-patch BASE_DIR so models download to D: not C:\Users\...\.paddleocr
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
    Lazy load PaddleOCR. Re-initializes if the target language changes.
    lang options: 'ch', 'en', 'korean', 'japan', etc.
    """
    global _paddle_ocr_instance, _paddle_last_lang
    if PaddleOCR is None:
        raise ImportError("PaddleOCR could not be loaded. Please ensure paddlepaddle-gpu and paddleocr are installed.")
        
    if _paddle_ocr_instance is None or _paddle_last_lang != lang:
        logger.info(f"Initializing PaddleOCR for language: {lang}")
        # Disable GPU for OCR because it causes WinError 126 missing cuDNN dlls, 
        # and we need to save GPU VRAM for the heavy SAM/LaMa models anyway.
        # CPU OCR on small text crops is extremely fast.
        _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=False)
        _paddle_last_lang = lang
        
    return _paddle_ocr_instance

def map_lang_to_paddle(lang_code: str) -> str:
    """
    Maps NLLB/ISO language codes (e.g. jpn_Jpan, EN, JP, fra_Latn) to PaddleOCR language codes.
    Supported by PaddleOCR: ch, en, korean, japan, fr, german, es, ru, it, pt, ar, hi, etc.
    """
    lang_code = lang_code.lower()
    
    # CJK
    if "jp" in lang_code or "japan" in lang_code: return "japan"
    if "ko" in lang_code or "kor" in lang_code: return "korean"
    if "zh" in lang_code or "ch" in lang_code:
        if "hant" in lang_code or "trad" in lang_code: return "chinese_cht"
        return "ch"
    
    # Latin / European
    if "en" in lang_code or "eng" in lang_code: return "en"
    if "fr" in lang_code or "fra" in lang_code: return "fr"
    if "es" in lang_code or "spa" in lang_code: return "es"
    if "de" in lang_code or "ger" in lang_code: return "german"
    if "it" in lang_code or "ita" in lang_code: return "it"
    if "pt" in lang_code or "por" in lang_code: return "pt"
    if "nl" in lang_code or "nld" in lang_code or "dut" in lang_code: return "nl"
    if "sv" in lang_code or "swe" in lang_code: return "sv"
    if "da" in lang_code or "dan" in lang_code: return "da"
    if "fi" in lang_code or "fin" in lang_code: return "fi"
    if "no" in lang_code or "nor" in lang_code: return "no"
    if "pl" in lang_code or "pol" in lang_code: return "pl"
    if "cs" in lang_code or "ces" in lang_code or "cze" in lang_code: return "cs"
    if "sk" in lang_code or "slk" in lang_code: return "sk"
    if "ro" in lang_code or "ron" in lang_code: return "ro"
    if "hu" in lang_code or "hun" in lang_code: return "hu"
    if "hr" in lang_code or "hrv" in lang_code: return "hr"
    if "af" in lang_code or "afr" in lang_code: return "af"
    if "az" in lang_code or "aze" in lang_code: return "az"
    if "bs" in lang_code or "bos" in lang_code: return "bs"
    if "cy" in lang_code or "cym" in lang_code or "wel" in lang_code: return "cy"
    if "et" in lang_code or "est" in lang_code: return "et"
    if "ga" in lang_code or "gle" in lang_code or "iri" in lang_code: return "ga"
    if "is" in lang_code or "isl" in lang_code or "ice" in lang_code: return "is"
    if "ku" in lang_code or "kur" in lang_code: return "ku"
    if "la" in lang_code or "lat" in lang_code: return "la"
    if "lt" in lang_code or "lit" in lang_code: return "lt"
    if "lv" in lang_code or "lav" in lang_code: return "lv"
    if "mi" in lang_code or "mri" in lang_code or "mao" in lang_code: return "mi"
    if "mt" in lang_code or "mlt" in lang_code: return "mt"
    if "sl" in lang_code or "slv" in lang_code: return "sl"
    if "sq" in lang_code or "sqi" in lang_code or "alb" in lang_code: return "sq"
    if "sw" in lang_code or "swa" in lang_code: return "sw"
    if "tl" in lang_code or "tgl" in lang_code: return "tl"
    if "uz" in lang_code or "uzb" in lang_code: return "uz"
    
    # Cyrillic
    if "ru" in lang_code or "rus" in lang_code: return "ru"
    if "bg" in lang_code or "bul" in lang_code: return "bg"
    if "uk" in lang_code or "ukr" in lang_code: return "uk"
    if "be" in lang_code or "bel" in lang_code: return "be"
    if "sr" in lang_code or "srp" in lang_code: return "rs_cyrillic"
    
    # Middle East / Central Asia
    if "ar" in lang_code or "ara" in lang_code: return "ar"
    if "ur" in lang_code or "urd" in lang_code: return "ur"
    if "fa" in lang_code or "pes" in lang_code or "per" in lang_code: return "fa"
    if "tr" in lang_code or "tur" in lang_code: return "tr"
    if "ug" in lang_code or "uig" in lang_code: return "ug"
    
    # Indic
    if "hi" in lang_code or "hin" in lang_code: return "hi"
    if "bn" in lang_code or "ben" in lang_code: return "bn"
    if "gu" in lang_code or "guj" in lang_code: return "gu"
    if "ta" in lang_code or "tam" in lang_code: return "ta"
    if "te" in lang_code or "tel" in lang_code: return "te"
    if "kn" in lang_code or "kan" in lang_code: return "ka"
    if "ml" in lang_code or "mal" in lang_code: return "ml"
    if "mr" in lang_code or "mar" in lang_code: return "mr"
    if "ne" in lang_code or "nep" in lang_code: return "ne"
    
    # SEA
    if "vi" in lang_code or "vie" in lang_code: return "vi"
    if "id" in lang_code or "ind" in lang_code: return "id"
    if "ms" in lang_code or "zsm" in lang_code or "msa" in lang_code: return "ms"
    
    # Default fallback
    return "japan"

def run_paddle_ocr_on_regions(image: Image.Image, regions: list, cfg: dict) -> list:
    """
    Given an image and a list of BubbleRegion objects (from MIT detector),
    crops each region and runs PaddleOCR on the crop to replace the text.
    """
    if not regions:
        return regions
        
    # Get the source language hint from config (default to japan if auto/unknown)
    source_lang_hint = cfg.get("source_lang_override") or "japan"
    paddle_lang = map_lang_to_paddle(source_lang_hint)
    
    ocr = get_paddle_ocr(paddle_lang)
    img_np = np.array(image.convert("RGB"))
    # OpenCV expects BGR
    img_np = img_np[:, :, ::-1] 
    
    for region in regions:
        # BubbleRegion provides bbox as (x1, y1, x2, y2)
        x1, y1, x2, y2 = region.bbox
        
        # Add a small padding to the crop
        pad = 5
        h, w = img_np.shape[:2]
        crop_y1, crop_y2 = max(0, int(y1)-pad), min(h, int(y2)+pad)
        crop_x1, crop_x2 = max(0, int(x1)-pad), min(w, int(x2)+pad)
        
        crop = img_np[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
            
        # Run PaddleOCR
        # result format: [[[[x,y], [x,y]...], ('text', confidence)], ...]
        result = ocr.ocr(crop, cls=True)
        
        if result and result[0]:
            # result[0] is a list of lines detected in the crop
            # Each line: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)]
            line_entries = []
            for line in result[0]:
                if len(line) >= 2 and len(line[1]) >= 1:
                    text = line[1][0]
                    box = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    x_center = sum(pt[0] for pt in box) / len(box)
                    y_center = sum(pt[1] for pt in box) / len(box)
                    # Calculate bounding box width and height to detect vertical text
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    box_w = max(xs) - min(xs)
                    box_h = max(ys) - min(ys)
                    line_entries.append((x_center, y_center, text, box_w, box_h))
            
            if line_entries:
                # Detect if the text is vertical: lines are taller than wide
                is_cjk = paddle_lang in ("japan", "korean", "ch", "chinese_cht")
                avg_aspect = sum(e[4] / max(e[3], 1) for e in line_entries) / len(line_entries)
                is_vertical = is_cjk and avg_aspect > 1.5

                if is_cjk and is_vertical:
                    # Vertical CJK text reads columns right-to-left, top-to-bottom
                    # Group X by ~15 pixels to handle slight tilts in the same column
                    line_entries.sort(key=lambda e: (-round(e[0] / 15), e[1]))
                else:
                    # Horizontal text (English, Horizontal CJK) reads top-to-bottom, left-to-right
                    # Group Y by ~15 pixels to handle slight tilts on the same line
                    line_entries.sort(key=lambda e: (round(e[1] / 15), e[0]))
                
                lines = [entry[2] for entry in line_entries]
                new_text = "\n".join(lines)
                logger.info(f"PaddleOCR replaced text '{region.source_text}' with '{new_text}' (vertical={is_vertical if is_cjk else 'n/a'})")
                region.source_text = new_text
                
    return regions
