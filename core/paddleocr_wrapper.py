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
            gpu_mem=500,
            det_db_thresh=0.15,      # More sensitive to faint/jagged text
            det_db_box_thresh=0.3,    # Lower threshold for box creation
            det_db_unclip_ratio=2.0   # Slightly larger boxes to catch outlines
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

def run_paddle_gap_filling(image: Image.Image, existing_regions: list, cfg: dict) -> list:
    """
    Scans the entire image for text. 
    If text is found that is NOT already inside an existing region (YOLO bubble), 
    returns a list of new suggested regions (x, y, w, h, text).
    """
    source_lang_hint = cfg.get("source_lang_override") or "japan"
    paddle_lang = map_lang_to_paddle(source_lang_hint)
    
    ocr = get_paddle_ocr(paddle_lang)
    img_np = np.array(image.convert("RGB"))
    img_np = img_np[:, :, ::-1] # BGR
    
    # Run global OCR
    result = ocr.ocr(img_np, cls=True)
    if not result or not result[0]:
        return []
        
    new_regions = []
    
    # Pre-calculate existing bounding boxes
    existing_bboxes = [r.bbox for r in existing_regions]
    
    for line in result[0]:
        if len(line) < 2: continue
        box = line[0]
        text = line[1][0]
        score = line[1][1]
        
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        w, h = x2 - x1, y2 - y1

        # ── Adaptive Threshold Logic ──────────────────────────────────────
        # Check background complexity to distinguish bubbles from SFX
        try:
            # Crop the area and check variance
            crop = image.crop((x1, y1, x2, y2)).convert("L")
            stat = np.array(crop)
            variance = np.std(stat)
            
            # Simple/Bubble/Transparent backgrounds have lower variance
            # Busy Art/SFX backgrounds have high variance
            if variance < 40:
                min_score = 0.15  # Very sensitive for bubbles/transparent areas
            else:
                # Use user-defined strictness for busy backgrounds
                min_score = cfg.get("sfx_strictness", 0.55)
        except:
            min_score = 0.3
            
        if score < min_score: continue 
        
        # Check if this box is already inside an existing region
        is_inside = False
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        for ex1, ey1, ex2, ey2 in existing_bboxes:
            if (ex1 - 10 <= box_center_x <= ex2 + 10) and (ey1 - 10 <= box_center_y <= ey2 + 10):
                is_inside = True
                break
        
        if not is_inside:
            pad = 4
            new_regions.append({
                "x": max(0, x1 - pad),
                "y": max(0, y1 - pad),
                "w": w + pad * 2,
                "h": h + pad * 2,
                "text": text
            })

    # ── Merge nearby new regions into "Virtual Bubbles" ─────────────────────
    if not new_regions:
        return []

    # Simple vertical/horizontal proximity merge
    merged_new = []
    new_regions.sort(key=lambda r: (r["y"], r["x"]))
    
    curr = new_regions[0]
    for nxt in new_regions[1:]:
        # If very close (overlapping or within 20px), merge them
        dist_y = nxt["y"] - (curr["y"] + curr["h"])
        overlap_x = min(curr["x"] + curr["w"], nxt["x"] + nxt["w"]) - max(curr["x"], nxt["x"])
        
        if dist_y < 25 and overlap_x > 0:
            # Merge
            x1 = min(curr["x"], nxt["x"])
            y1 = min(curr["y"], nxt["y"])
            x2 = max(curr["x"] + curr["w"], nxt["x"] + nxt["w"])
            y2 = max(curr["y"] + curr["h"], nxt["y"] + nxt["h"])
            curr = {
                "x": x1, "y": y1, "w": x2-x1, "h": y2-y1,
                "text": (curr["text"] + " " + nxt["text"]).strip()
            }
        else:
            merged_new.append(curr)
            curr = nxt
    merged_new.append(curr)

    return merged_new
