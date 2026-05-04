import os
import numpy as np
import logging
from PIL import Image, ImageOps

logger = logging.getLogger("core.easyocr")

# ── Cache Directory ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EASY_CACHE = os.path.join(_ROOT, "model", "easyocr_cache")
os.makedirs(_EASY_CACHE, exist_ok=True)

_easy_reader_instance = None
_easy_last_langs = None

def get_easyocr_reader(langs=["ja", "en"]):
    """
    Lazy load EasyOCR Reader. Re-initializes if the target languages change.
    """
    global _easy_reader_instance, _easy_last_langs
    
    try:
        import easyocr
    except ImportError:
        logger.error("EasyOCR pip package not found. Please run 'pip install easyocr'")
        return None
        
    # Sort to ensure consistent cache key
    langs = sorted(list(set(langs)))
    
    if _easy_reader_instance is None or _easy_last_langs != langs:
        logger.info(f"Initializing EasyOCR for languages: {langs}")
        # EasyOCR constructor is slow and uses VRAM
        _easy_reader_instance = easyocr.Reader(
            langs, 
            gpu=True, 
            model_storage_directory=_EASY_CACHE,
            download_enabled=True
        )
        _easy_last_langs = langs
        
    return _easy_reader_instance

def map_lang_to_easy(lang_code: str) -> list:
    """
    Maps ISO/NLLB language codes to EasyOCR language codes.
    Returns a list because EasyOCR supports multi-language OCR.
    """
    if not lang_code:
        return ["ja", "en"]
    lang_code = lang_code.lower()
    if "jp" in lang_code or "japan" in lang_code: return ["ja", "en"]
    if "ko" in lang_code or "kor" in lang_code: return ["ko", "en"]
    if "zh" in lang_code or "ch" in lang_code:
        if "hant" in lang_code or "trad" in lang_code: return ["ch_tra", "en"]
        return ["ch_sim", "en"]
    if "en" in lang_code or "eng" in lang_code: return ["en"]
    if "fr" in lang_code or "fra" in lang_code: return ["fr", "en"]
    if "es" in lang_code or "spa" in lang_code: return ["es", "en"]
    if "de" in lang_code or "ger" in lang_code: return ["de", "en"]
    
    # Default fallback
    return ["ja", "en"]

def run_easyocr_on_regions(image: Image.Image, regions: list, cfg: dict = None) -> list:
    """
    Runs EasyOCR on a list of BubbleRegion objects.
    Modifies regions in-place.
    """
    if not regions:
        return regions
        
    lang_code = cfg.get("source_lang", "eng_Latn") if cfg else "eng_Latn"
    easy_langs = map_lang_to_easy(lang_code)
    # Use print so it definitely shows up in the user's CMD
    print(f"DEBUG: [EasyOCR] Runtime Config - Source: {lang_code} -> EasyLangs: {easy_langs}")
    
    # Force reset if language changed
    global _easy_reader_instance, _easy_last_langs
    if _easy_last_langs != sorted(list(set(easy_langs))):
        _easy_reader_instance = None
    
    reader = get_easyocr_reader(easy_langs)
    if reader is None:
        return regions
        
    img_np = np.array(image.convert("RGB"))
    w_orig, h_orig = image.size
    
    super_res = cfg.get("ocr_super_res", True) if cfg else True
    upscale_factor = float(cfg.get("ocr_upscale_factor", 2.0)) if cfg else 2.0

    for region in regions:
        x1, y1, x2, y2 = region.bbox
        # Aggressive padding for comic fonts
        pad = 10
        crop_y1, crop_y2 = max(0, int(y1)-pad), min(h_orig, int(y2)+pad)
        crop_x1, crop_x2 = max(0, int(x1)-pad), min(w_orig, int(x2)+pad)
        
        crop_pil = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("L")
        if crop_pil.size[0] < 2 or crop_pil.size[1] < 2:
            continue
            
        if super_res:
            # Upscale for better recognition on low-quality/fuzzy images
            w, h = crop_pil.size
            crop_pil = crop_pil.resize((int(w*upscale_factor), int(h*upscale_factor)), resample=Image.LANCZOS)
            
            # Simple contrast boost
            crop_pil = ImageOps.autocontrast(crop_pil, cutoff=2)
            
        crop = np.array(crop_pil.convert("RGB"))
            
        try:
            # detail=1 returns (bbox, text, confidence)
            results = reader.readtext(crop, detail=1, paragraph=False)
            if results:
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                region.source_text = " ".join(texts).strip()
                region.confidence = sum(confs) / len(confs)
        except Exception as e:
            logger.warning(f"EasyOCR failed on a region: {e}")
                
    return regions
