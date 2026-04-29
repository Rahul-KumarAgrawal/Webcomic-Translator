import os
import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ── Auto-Detect Tesseract Path ──────────────────────────────────────────────
_COMMON_PATHS = [
    r"Tesseract-OCR\tesseract.exe",  # Local project folder
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Users\%USERNAME%\AppData\Local\Tesseract-OCR\tesseract.exe",
    r"D:\Tesseract-OCR\tesseract.exe",
]

def _find_tesseract():
    # 1. Check if it's already in PATH
    import shutil
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd
    
    # 2. Check common install locations
    for p in _COMMON_PATHS:
        expanded = os.path.expandvars(p)
        if os.path.exists(expanded):
            return expanded
    return None

# Set the command path
tesseract_path = _find_tesseract()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logger.info(f"[Tesseract] Found engine at: {tesseract_path}")
else:
    logger.warning("[Tesseract] Engine not found. Please install Tesseract-OCR.")

def map_lang_to_tess(lang_code: str) -> str:
    """Maps NLLB/ISO codes to Tesseract .traineddata codes."""
    lang_code = lang_code.lower()
    if "jp" in lang_code: return "jpn"
    if "ko" in lang_code: return "kor"
    if "zh" in lang_code:
        if "hant" in lang_code: return "chi_tra"
        return "chi_sim"
    if "en" in lang_code: return "eng"
    if "es" in lang_code: return "spa"
    if "fr" in lang_code: return "fra"
    if "de" in lang_code: return "deu"
    return "eng" # Fallback

def run_tesseract_on_regions(image, regions, cfg=None):
    """
    Runs Tesseract OCR on a list of BubbleRegion objects.
    Modifies regions in-place.
    """
    if not tesseract_path:
        return regions

    lang_code = cfg.get("source_lang_override") if cfg else "eng_Latn"
    tess_lang = map_lang_to_tess(lang_code)
    
    # ... (rest of language verification) ...
    tess_dir = os.path.dirname(tesseract_path)
    tessdata_path = os.path.join(tess_dir, "tessdata", f"{tess_lang}.traineddata")
    
    if not os.path.exists(tessdata_path) and tess_lang != "eng":
        logger.error(f"[Tesseract] MISSING LANGUAGE DATA: {tessdata_path} not found.")
        logger.error(f"[Tesseract] Please download {tess_lang}.traineddata from https://github.com/tesseract-ocr/tessdata")
        logger.error("[Tesseract] Falling back to 'eng' (expect poor results).")
        tess_lang = "eng"

    logger.info(f"[Tesseract] Running OCR with lang={tess_lang} (Source: {lang_code})")
    
    super_res = cfg.get("ocr_super_res", True) if cfg else True
    upscale_factor = float(cfg.get("ocr_upscale_factor", 2.0)) if cfg else 2.0

    for r in regions:
        # Crop the region with a small extra padding for Tesseract
        pad = 5
        crop = image.crop((max(0, r.x-pad), max(0, r.y-pad), min(image.width, r.x+r.w+pad), min(image.height, r.y+r.h+pad)))
        
        if super_res:
            # ── Pre-OCR Upscaling (Crucial for Tesseract on low-res) ──
            from PIL import Image, ImageOps
            w_c, h_c = crop.size
            crop = crop.resize((int(w_c*upscale_factor), int(h_c*upscale_factor)), resample=Image.LANCZOS)
            crop = ImageOps.autocontrast(crop.convert("L"), cutoff=2)
        
        try:
            # psm 6 is usually best for speech bubbles (Assume a single uniform block of text)
            # For CJK, sometimes psm 3 or 11 works too, but 6 is the stable choice.
            text = pytesseract.image_to_string(crop, lang=tess_lang, config='--psm 6')
            r.source_text = text.strip()
        except Exception as e:
            logger.error(f"[Tesseract] Error on region {r}: {e}")
            
    return regions
