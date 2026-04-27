"""
core/batch_processor.py
CLI batch processing manager.

Usage (via run_batch.bat):
    python core/batch_processor.py --input ./input --output ./output --series "Solo Leveling"

Processes all CBZ files in --input sequentially, skipping already-translated files.
After processing, triggers fine-tuning if ≥ 10 approved pairs exist.
Sends a Windows toast notification on completion.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.cbz_handler import extract_cbz, repack_cbz
from core.inpainter import Inpainter
from core.notifier import notify_batch_done, notify_finetune_done
from core.translator import Translator
from core.google_vision import GoogleVisionDetector, GOOGLE_TO_NLLB
from memory.memory_manager import MemoryManager

# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{stamp}_session.log")

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logger = logging.getLogger("batch_processor")
    logger.info("Session log: %s", log_path)
    return logger


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_cfg(cfg_path: str = None) -> dict:
    import yaml
    if cfg_path is None:
        cfg_path = os.path.join(_ROOT, "config", "settings.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[WARNING] Could not load settings.yaml: {exc}. Using defaults.")
        return {}


def _get_font_cfg(cfg: dict, series: str) -> dict:
    """Get merged font config for a series (series overrides global default)."""
    default = (cfg.get("default_font") or {
        "family": None, "color": "auto", "size": "auto"
    })
    overrides = (cfg.get("series_fonts") or {}).get(series, {})
    return {**default, **overrides}


# ── Auto-Detect Language ────────────────────────────────────────────────────────

LANGDETECT_TO_NLLB = {
    "ja": "jpn_Jpan",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ko": "kor_Hang",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "sk": "slk_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "hr": "hrv_Latn",
    "ru": "rus_Cyrl",
    "bg": "bul_Cyrl",
    "uk": "ukr_Cyrl",
    "be": "bel_Cyrl",
    "sr": "srp_Cyrl",
    "ar": "arb_Arab",
    "fa": "pes_Arab",
    "ur": "urd_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "ne": "nep_Deva",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "af": "afr_Latn",
    "sq": "sqi_Latn",
    "sw": "swa_Latn",
    "tl": "tgl_Latn",
}

def _auto_detect_language(images: list, logger: logging.Logger, cfg: dict = None) -> Tuple[Optional[str], float]:
    """
    Identifies the language of the comic.
    Respects 'auto_detect_engine' from settings.
    """
    sample_pages = images[:min(3, len(images))]
    engine = cfg.get("auto_detect_engine", "gemini") if cfg else "gemini"

    # ── Choice 1: Gemini Multimodal (Recommended) ───────────────────────────
    if engine == "gemini":
        google_key = cfg.get("google_api_key") if cfg else None
        if google_key:
            logger.info("  Auto-detect: using Google Gemini Multimodal API...")
            try:
                from core.google_translator import GoogleTranslator
                detector = GoogleTranslator(api_key=google_key)
                test_img = sample_pages[1] if len(sample_pages) > 1 else sample_pages[0]
                detected = detector.detect_language_from_image(test_img)
                if detected and detected != "unknown":
                    logger.info("  Auto-detect: Gemini detected '%s'", detected)
                    return detected, 1.0
            except Exception as e:
                logger.warning("  Gemini auto-detect failed: %s.", e)
        else:
            logger.warning("  Auto-detect: Gemini selected but no API key found.")

    # ── Choice 2: Google Cloud Vision API ────────────────────────────────────
    if engine == "google":
        vision_key = os.path.join(_ROOT, "config", "google_vision_key.json")
        if os.path.exists(vision_key):
            logger.info("  Auto-detect: using Google Cloud Vision API...")
            try:
                detector = GoogleVisionDetector(vision_key)
                test_img = sample_pages[1] if len(sample_pages) > 1 else sample_pages[0]
                detected = detector.detect_language(test_img)
                if detected:
                    logger.info("  Auto-detect: Google Vision detected '%s'", detected)
                    return detected, 1.0
            except Exception as e:
                logger.warning("  Google Vision auto-detect failed: %s.", e)
        else:
            logger.warning("  Auto-detect: Google Cloud Vision selected but no JSON key found in config/.")

    # ── Choice 3: Fallback / Manual Selection: Local EasyOCR ─────────────────
    if engine == "local" or True: # Fallback to local if others fail
        if engine != "local":
            logger.info("  Auto-detect: falling back to local OCR...")
        
        try:
            import easyocr
            from langdetect import detect_langs
        except ImportError:
            logger.error("easyocr or langdetect not installed. Cannot auto-detect language.")
            return None, 0.0

        easyocr_dir = os.path.join(_ROOT, "models", "easyocr")
        os.makedirs(easyocr_dir, exist_ok=True)
        _kwargs = dict(gpu=True, model_storage_directory=easyocr_dir, verbose=False)

        # Language groups ordered by likelihood for manga/manhwa/manhua.
        LANG_GROUPS = [
            (['ja', 'en'],                          "Japanese + English"),
            (['ko', 'en'],                          "Korean + English"),
            (['ch_sim', 'en'],                      "Chinese (Simplified) + English"),
            (['ch_tra', 'en'],                      "Chinese (Traditional) + English"),
        ]

        best_overall_lang = None
        best_overall_prob = 0.0

        for lang_list, group_name in LANG_GROUPS:
            logger.info("  Auto-detect: trying %s ...", group_name)
            try:
                reader = easyocr.Reader(lang_list, **_kwargs)
            except Exception as e:
                logger.warning("  Skipping group %s: %s", group_name, e)
                continue

            # Run OCR on sample pages with this reader
            group_text = ""
            for img_path in sample_pages:
                try:
                    results = reader.readtext(img_path, detail=0)
                    group_text += "\n" + "\n".join(results)
                except Exception:
                    pass

            # Free this reader before potentially loading the next one
            del reader
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Check this group's text independently
            if len(group_text.strip()) > 15:
                try:
                    langs = detect_langs(group_text)
                    best_lang = langs[0]
                    logger.info("  Auto-detect result for %s: %s (%.1f%%)",
                                group_name, best_lang.lang, best_lang.prob * 100)
                    
                    if best_lang.prob > best_overall_prob:
                        best_overall_prob = best_lang.prob
                        best_overall_lang = best_lang.lang

                    if best_overall_prob >= 0.90:
                        break
                except Exception:
                    pass

    if best_overall_prob >= 0.90:
        return best_overall_lang, best_overall_prob

    # Fallback: take the best guess if it's somewhat confident
    if best_overall_lang and best_overall_prob > 0.50:
        logger.info("  Auto-detect fallback guess: %s (%.1f%%)",
                    best_overall_lang, best_overall_prob * 100)
        return best_overall_lang, best_overall_prob

    logger.warning("  Could not detect language confidently (max confidence %.1f%%).", best_overall_prob * 100)
    return None, best_overall_prob


# ── Main batch logic ───────────────────────────────────────────────────────────

def process_cbz(
    cbz_path: str,
    output_dir: str,
    series: str,
    cfg: dict,
    logger: logging.Logger,
    progress_callback=None,
) -> dict:
    """
    Translate a single CBZ file.

    Returns a dict with:
      { success, output_path, num_bubbles, elapsed }
    """
    start_time = time.time()
    cbz_name   = Path(cbz_path).name
    output_cbz = os.path.join(output_dir, cbz_name)

    # Skip if already translated (unless force_retranslate is set)
    force = cfg.get("force_retranslate", False)
    if os.path.exists(output_cbz) and not force:
        logger.info("SKIP (already translated): %s", cbz_name)
        return {"success": True, "output_path": output_cbz, "num_bubbles": 0,
                "elapsed": 0, "skipped": True}

    # If forcing, remove prior output + session data so we start fresh
    if force and os.path.exists(output_cbz):
        logger.info("FORCE RE-TRANSLATE: removing old output for %s", cbz_name)
        os.remove(output_cbz)
    session_path = os.path.join(_ROOT, "web", "sessions",
                                cbz_name.replace(".cbz", "") + ".json")
    if force and os.path.exists(session_path):
        os.remove(session_path)
    crops_dir_old = os.path.join(_ROOT, "web", "sessions", "crops",
                                 cbz_name.replace(".cbz", ""))
    if force and os.path.exists(crops_dir_old):
        shutil.rmtree(crops_dir_old, ignore_errors=True)

    logger.info("━━━ Processing: %s ━━━", cbz_name)
    tmp_dir = None

    # ── Branch: MIT full pipeline vs. existing pipeline ────────────────────────
    use_mit = cfg.get("use_mit_pipeline", False)
    use_koharu = cfg.get("use_koharu_pipeline", False)

    try:
        # 1. Extract CBZ (with optional chunking)
        webtoon_strip_height = int(cfg.get("webtoon_strip_height", 0))
        chunk_height = int(cfg.get("chunk_height", 0))
        chunk_overlap = int(cfg.get("chunk_overlap", 0))
        tmp_dir, images = extract_cbz(
            cbz_path,
            webtoon_strip_height=webtoon_strip_height,
            chunk_height=chunk_height,
            chunk_overlap=chunk_overlap,
        )
        total_pages = len(images)
        logger.info("  Pages/Chunks: %d", total_pages)

        # ── Language Auto-Detect Phase ─────────────────────────────────────────
        source_lang_override = cfg.get("source_lang_override")
        if not source_lang_override or source_lang_override == "auto":
            logger.info("Source language is set to Auto-Detect. Starting detection phase...")
            detected_iso, prob = _auto_detect_language(images, logger, cfg)
            if not detected_iso or prob < 0.50:
                return {
                    "success": False,
                    "error": "Could not extract enough text to auto-detect language. Please select the language manually."
                }
            
            nllb_code = LANGDETECT_TO_NLLB.get(detected_iso)
            if not nllb_code:
                return {
                    "success": False,
                    "error": f"Detected language ISO code '{detected_iso}' is not supported by the translation pipeline. Please select manually."
                }
            
            # Map NLLB to readable name for the warning
            readable_lang = {
                "jpn_Jpan": "Japanese",
                "zho_Hans": "Chinese (Simplified)",
                "zho_Hant": "Chinese (Traditional)",
                "kor_Hang": "Korean",
                "eng_Latn": "English",
            }.get(nllb_code, nllb_code.split('_')[0].capitalize())

            if prob < 0.90:
                return {
                    "success": False,
                    "error": f"Language detected as {readable_lang} ({prob*100:.0f}% confidence). If this looks wrong, please select manually."
                }

            logger.info(f"Auto-detection successful. Confirmed source language: {nllb_code}")
            cfg["source_lang_override"] = nllb_code

        # Load chunk metadata if chunking was used (for deduplication)
        chunk_meta = None
        chunk_meta_path = os.path.join(tmp_dir, "chunk_metadata.json") if tmp_dir else None
        if chunk_meta_path and os.path.exists(chunk_meta_path):
            import json as _json
            with open(chunk_meta_path, "r", encoding="utf-8") as _f:
                chunk_meta = _json.load(_f)
            logger.info("  Chunk dedup enabled: %d chunks, overlap=%dpx",
                        chunk_meta["num_chunks"], chunk_meta["chunk_overlap"])

        output_tmp = tempfile.mkdtemp(prefix="cbz_output_")
        all_bubble_results = []  # list of (BubbleResult, page_num, crop_url)

        # Directory for bubble crop images and blank background cache
        crops_dir = os.path.join(_ROOT, "web", "sessions", "crops",
                                 cbz_name.replace(".cbz", ""))
        bg_cache_dir = os.path.join(_ROOT, "web", "sessions", "bg_cache",
                                    cbz_name.replace(".cbz", ""))
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(bg_cache_dir, exist_ok=True)

        if use_koharu:
            # ── KOHARU FULL PIPELINE ───────────────────────────────────────────
            all_bubble_results = _process_pages_koharu(
                images, output_tmp, crops_dir, bg_cache_dir, cbz_name,
                cfg, logger, progress_callback, total_pages,
                chunk_meta=chunk_meta,
            )
        elif use_mit:
            # ── MIT FULL PIPELINE ──────────────────────────────────────────────
            all_bubble_results = _process_pages_mit(
                images, output_tmp, crops_dir, cbz_name,
                cfg, logger, progress_callback, total_pages,
                chunk_meta=chunk_meta,
            )
        else:
            # ── STANDARD MODULAR PIPELINE ──────────────────────────────────────
            all_bubble_results = _process_pages_standard(
                images, output_tmp, crops_dir, bg_cache_dir, cbz_name, series,
                cfg, logger, progress_callback, total_pages,
                chunk_meta=chunk_meta,
            )

        # 5. Reassemble chunks: trim overlap so output has no duplicated content
        if chunk_meta and chunk_meta.get("chunk_overlap", 0) > 0:
            _reassemble_chunks(output_tmp, chunk_meta, logger, all_bubble_results)

        # 6. Repack translated images into output CBZ
        os.makedirs(output_dir, exist_ok=True)
        repack_cbz(output_tmp, output_cbz)
        logger.info("  ✓ Output CBZ: %s", output_cbz)

        # 6. Save session bubble data for review page
        _save_session_data(cbz_name, all_bubble_results, series, chunk_meta=chunk_meta)

        elapsed = time.time() - start_time
        return {
            "success":     True,
            "output_path": output_cbz,
            "num_bubbles": len(all_bubble_results),
            "elapsed":     elapsed,
        }

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("FAILED %s: %s", cbz_name, exc, exc_info=True)
        return {
            "success": False,
            "error":   str(exc),
            "elapsed": elapsed,
        }
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Chunk reassembly (trim overlap for clean output) ────────────────────────

def _reassemble_chunks(output_dir: str, chunk_meta: dict, logger: logging.Logger, all_bubble_results: list = None):
    """
    After translating overlapping chunks, trim the overlap regions so the
    final output has no duplicated content.

    Strategy: for each pair of adjacent chunks, split the overlap zone at
    its midpoint. The earlier chunk keeps its top portion (up to the mid-
    point), and the later chunk keeps its bottom portion (from the midpoint
    down). This gives the cleanest seam because each bubble is fully
    contained in the chunk where it appeared most completely.
    """
    from PIL import Image

    overlap = chunk_meta.get("chunk_overlap", 0)
    chunks_info = chunk_meta.get("chunks", [])
    num_chunks = len(chunks_info)

    if num_chunks <= 1 or overlap <= 0:
        return

    # Helper: Pre-calculate safe seams for each overlap zone
    def _find_safe_seam(h_A: int, actual_overlap: int, bubbles_A: list, bubbles_B: list) -> int:
        if actual_overlap <= 0:
            return 0
            
        # Try decreasing padding until we find a gap. 
        # Large padding (50) ensures we never cut through overflowing English text.
        for padding in [50, 30, 15, 0]:
            intervals = []
            
            for region in bubbles_A:
                y_start = region.y - (h_A - actual_overlap) - padding
                y_end = (region.y + region.h) - (h_A - actual_overlap) + padding
                if y_end > 0 and y_start < actual_overlap:
                    intervals.append([max(0, y_start), min(actual_overlap, y_end)])
                    
            for region in bubbles_B:
                y_start = region.y - padding
                y_end = (region.y + region.h) + padding
                if y_end > 0 and y_start < actual_overlap:
                    intervals.append([max(0, y_start), min(actual_overlap, y_end)])
                    
            if not intervals:
                return actual_overlap // 2
                
            intervals.sort(key=lambda x: x[0])
            merged = []
            for iv in intervals:
                if not merged:
                    merged.append(iv)
                else:
                    last = merged[-1]
                    if iv[0] <= last[1]:
                        last[1] = max(last[1], iv[1])
                    else:
                        merged.append(iv)
                        
            gaps = []
            if merged[0][0] > 0:
                gaps.append((0, merged[0][0]))
            for i in range(len(merged) - 1):
                gaps.append((merged[i][1], merged[i+1][0]))
            if merged[-1][1] < actual_overlap:
                gaps.append((merged[-1][1], actual_overlap))
                
            if gaps:
                best_gap = max(gaps, key=lambda g: g[1] - g[0])
                return (best_gap[0] + best_gap[1]) // 2
                
        # If no gaps found even with 0 padding, fallback to exact midpoint
        return actual_overlap // 2

    # Map bubbles to chunks
    bubbles_by_chunk = {i: [] for i in range(num_chunks)}
    if all_bubble_results:
        for br, region, page_num, crop_url in all_bubble_results:
            idx = page_num - 1
            if 0 <= idx < num_chunks:
                bubbles_by_chunk[idx].append(region)

    seams = []
    for idx in range(num_chunks - 1):
        cinfo_A = chunks_info[idx]
        cinfo_B = chunks_info[idx + 1]
        
        h_A = cinfo_A["y_end"] - cinfo_A["y_start"]
        actual_overlap = cinfo_A["y_end"] - cinfo_B["y_start"]
        
        if actual_overlap <= 0:
            seams.append(0)
            continue
            
        bubbles_A = bubbles_by_chunk[idx]
        bubbles_B = bubbles_by_chunk[idx + 1]
        seam = _find_safe_seam(h_A, actual_overlap, bubbles_A, bubbles_B)
        seams.append(seam)

    logger.info("Reassembling %d chunks using dynamic seam carving...", num_chunks)

    reassembled_paths = []

    for idx, cinfo in enumerate(chunks_info):
        chunk_file = cinfo["file"]
        # Translated output uses the same filename pattern in output_dir
        out_name = f"page_{idx + 1:04d}.png"
        out_path = os.path.join(output_dir, out_name)

        if not os.path.exists(out_path):
            # Try chunk naming pattern
            out_name = chunk_file
            out_path = os.path.join(output_dir, out_name)
        if not os.path.exists(out_path):
            logger.warning("  Reassemble: cannot find output for chunk %d, skipping", idx)
            continue

        try:
            with Image.open(out_path) as img:
                w, h = img.size

                # Determine crop boundaries based on actual global coordinates
                crop_top = 0
                crop_bottom = h

                if idx > 0:
                    # Trim top half of the overlap with the previous chunk
                    prev_cinfo = chunks_info[idx - 1]
                    actual_overlap = prev_cinfo["y_end"] - cinfo["y_start"]
                    if actual_overlap > 0:
                        crop_top = seams[idx - 1]

                if idx < num_chunks - 1:
                    # Trim bottom half of the overlap with the next chunk
                    next_cinfo = chunks_info[idx + 1]
                    actual_overlap = cinfo["y_end"] - next_cinfo["y_start"]
                    if actual_overlap > 0:
                        crop_bottom = h - (actual_overlap - seams[idx])

                if crop_top == 0 and crop_bottom == h:
                    continue

                cropped = img.crop((0, crop_top, w, crop_bottom))
                cropped.save(out_path, format="PNG")
                logger.debug("  Chunk %d: trimmed y[%d:%d] → %dpx",
                             idx, crop_top, crop_bottom, crop_bottom - crop_top)
        except Exception as e:
            logger.warning("  Reassemble error on chunk %d: %s", idx, e)

    logger.info("Chunk reassembly complete — overlap trimmed from output.")


# ── Chunk deduplication helper ──────────────────────────────────────────────

def _is_duplicate_bubble(
    bubble_y: int,
    bubble_text: str,
    chunk_idx: int,
    chunk_meta: dict,
    seen_bubbles: list,
    proximity_px: int = 50,
    text_threshold: float = 0.3,
) -> bool:
    """
    Check if a bubble at *bubble_y* (in chunk-local coords) with *bubble_text*
    was already processed in a previous chunk's overlap region.

    *seen_bubbles* is a list of (global_y, text) from all prior chunks.
    Returns True if this bubble should be skipped (duplicate).
    """
    # Disable deduplication for chunks to prevent blank bubbles at seams.
    # We rely on _reassemble_chunks using a dynamic seam to crop the overlap cleanly.
    return False

    chunks = chunk_meta.get("chunks", [])
    if chunk_idx >= len(chunks):
        return False

    current_chunk = chunks[chunk_idx]
    overlap = chunk_meta.get("chunk_overlap", 0)

    # Only check bubbles in the overlap region (top of current chunk)
    if bubble_y > overlap:
        return False  # bubble is below overlap zone — definitely new

    # Convert to global y
    global_y = current_chunk["y_start"] + bubble_y

    # Check against previously seen bubbles
    try:
        import editdistance
        has_editdist = True
    except ImportError:
        has_editdist = False

    for prev_global_y, prev_text in seen_bubbles:
        # Position proximity check
        if abs(global_y - prev_global_y) > proximity_px:
            continue

        # Text similarity check
        if has_editdist:
            max_len = max(len(bubble_text), len(prev_text), 1)
            dist = editdistance.eval(bubble_text, prev_text)
            similarity = 1.0 - (dist / max_len)
        else:
            # Fallback: exact match or substring
            similarity = 1.0 if bubble_text == prev_text else (
                0.8 if (bubble_text in prev_text or prev_text in bubble_text) else 0.0
            )

        if similarity >= (1.0 - text_threshold):
            return True  # duplicate

    return False


def _bubble_to_global_y(bubble_y: int, chunk_idx: int, chunk_meta: dict) -> int:
    """Convert a chunk-local y coordinate to a global stitched y coordinate."""
    if not chunk_meta:
        return bubble_y
    chunks = chunk_meta.get("chunks", [])
    if chunk_idx >= len(chunks):
        return bubble_y
    return chunks[chunk_idx]["y_start"] + bubble_y


def _process_pages_standard(
    images, output_tmp, crops_dir, bg_cache_dir, cbz_name, series,
    cfg, logger, progress_callback, total_pages,
    chunk_meta=None,
):
    """Process all pages using the existing Inpainter + Translator pipeline."""
    from core.translator import Translator, BubbleResult

    translator = Translator(series=series, cfg=cfg)
    inpainter  = Inpainter(cfg=cfg)
    font_cfg   = _get_font_cfg(cfg, series)
    inpaint_engine = cfg.get("inpaint_engine", "lama")

    panelcleaner = None
    if inpaint_engine == "panelcleaner":
        from core.panelcleaner_wrapper import PanelCleanerPipeline
        device = cfg.get("gpu_device", "cuda")
        try:
            panelcleaner = PanelCleanerPipeline(device=device)
        except Exception as e:
            logger.error(f"Failed to load PanelCleaner models: {e}")
            raise

    all_bubble_results = []
    seen_bubbles = []  # list of (global_y, source_text) for dedup
    
    # We'll store intermediate results per page to process in phases
    # Phase Data: { page_num, img_path, regions, pil_image }
    page_data_list = []

    # ── Phase 1: OCR All Pages ───────────────────────────────────────────────
    logger.info("━━━ Phase 1: Extracting text from all pages...")
    for page_idx, img_path in enumerate(images):
        page_num = page_idx + 1
        logger.info("  OCR Page %d/%d: %s", page_num, total_pages, Path(img_path).name)
        try:
            image = Image.open(img_path).convert("RGB")
            _, regions = inpainter.detect_and_ocr(img_path)
            
            # Filter non-empty and non-duplicate regions immediately
            valid_regions_for_page = []
            for b_idx, region in enumerate(regions):
                if not region.source_text.strip(): continue
                if chunk_meta and _is_duplicate_bubble(region.y, region.source_text, page_idx, chunk_meta, seen_bubbles):
                    continue
                
                region.font_cfg = font_cfg
                valid_regions_for_page.append(region)
                
                if chunk_meta:
                    global_y = _bubble_to_global_y(region.y, page_idx, chunk_meta)
                    seen_bubbles.append((global_y, region.source_text))

            page_data_list.append({
                "page_num": page_num,
                "img_path": img_path,
                "regions": regions, # keep all for inpainting
                "valid_regions": valid_regions_for_page,
                "image": image
            })
        except Exception as e:
            logger.error("  OCR Error on page %d: %s", page_num, e)
            page_data_list.append({"page_num": page_num, "img_path": img_path, "regions": [], "valid_regions": [], "image": None})

        if progress_callback:
            # First half of progress for OCR
            progress_callback(page_idx + 1, total_pages * 2, text=f"Scanning Page {page_num}/{total_pages}")

    # ── Phase 2: Chapter-Wide Batch Translation ──────────────────────────────
    logger.info("━━━ Phase 2: Translating entire chapter batch...")
    all_texts_to_translate = []
    region_map = [] # To map back results: (page_data_idx, region_idx)

    for p_idx, p_data in enumerate(page_data_list):
        for r_idx, region in enumerate(p_data["valid_regions"]):
            all_texts_to_translate.append(region.source_text)
            region_map.append((p_idx, r_idx))

    if all_texts_to_translate:
        if progress_callback:
            progress_callback(total_pages, total_pages * 2, text=f"Translating {len(all_texts_to_translate)} bubbles...")

        if cfg.get("translation_engine") == "manual":
            logger.info("  Manual mode: skipping translation phase.")
            for p_data in page_data_list:
                for r in p_data["valid_regions"]: r.translated_text = ""
        else:
            # We translate in large chunks (e.g., 100 bubbles) to avoid hitting Gemini output limits
            # but still provide massive context.
            CHUNK_SIZE = 100
            all_translated_results = []
            
            for i in range(0, len(all_texts_to_translate), CHUNK_SIZE):
                chunk = all_texts_to_translate[i : i + CHUNK_SIZE]
                logger.info("  Translating batch chunk %d-%d/%d...", i+1, min(i+CHUNK_SIZE, len(all_texts_to_translate)), len(all_texts_to_translate))
                chunk_results = translator.translate_batch(chunk)
                all_translated_results.extend(chunk_results)

            # Apply results back to regions
            for (p_idx, r_idx), result in zip(region_map, all_translated_results):
                page_data_list[p_idx]["valid_regions"][r_idx].translated_text = result.translated_text
                # We'll store the result object for later too
                page_data_list[p_idx]["valid_regions"][r_idx]._result = result

    # ── Phase 3: Inpaint & Render All Pages ──────────────────────────────────
    logger.info("━━━ Phase 3: Inpainting and Rendering final pages...")
    for page_idx, p_data in enumerate(page_data_list):
        page_num = p_data["page_num"]
        img_path = p_data["img_path"]
        image = p_data["image"]
        regions = p_data["regions"]
        valid_regions = p_data["valid_regions"]

        logger.info("  Render Page %d/%d", page_num, total_pages)
        if not image: continue

        try:
            # Inpaint
            import numpy as np
            if inpaint_engine == "panelcleaner" and panelcleaner:
                np_img = np.array(image)
                mask = panelcleaner.detect_text_mask(np_img)
                inpainted_np = panelcleaner.inpaint_lama(np_img, mask)
                inpainted = Image.fromarray(inpainted_np)
            elif inpaint_engine in ("lama", "aot", "solid") and regions:
                inpainted = inpainter.inpaint(image, regions)
            else:
                inpainted = image.copy()

            bg_cache_path = os.path.join(bg_cache_dir, f"page_{page_num:04d}.png")
            inpainted.save(bg_cache_path, format="PNG")

            # Render
            out_page = os.path.join(output_tmp, Path(img_path).stem + ".png")
            final = inpainter.render_text(inpainted, regions)
            final.save(out_page, format="PNG", optimize=False)

            # Record results for review UI
            for b_idx, region in enumerate(valid_regions):
                crop_url = _save_crop(image, region, page_num, b_idx, crops_dir, cbz_name)
                
                if cfg.get("translation_engine") == "manual":
                    from core.translator import BubbleResult
                    res = BubbleResult(region.source_text, "", "", 0.0, "manual", False, False, None)
                else:
                    res = getattr(region, "_result", None)
                
                if res:
                    all_bubble_results.append((res, region, page_num, crop_url))

        except Exception as exc:
            logger.error("  Render Error on page %d: %s", page_num, exc)
            import shutil as _sh
            out_page = os.path.join(output_tmp, Path(img_path).name)
            _sh.copy2(img_path, out_page)

        if progress_callback:
            # Second half of progress for Rendering
            progress_callback(total_pages + page_idx + 1, total_pages * 2, text=f"Rendering Page {page_num}/{total_pages}")

    inpainter.unload_models()
    translator.unload_model()
    return all_bubble_results


def _process_pages_koharu(
    images, output_tmp, crops_dir, bg_cache_dir, cbz_name,
    cfg, logger, progress_callback, total_pages,
    chunk_meta=None,
):
    """Process all pages using the Koharu end-to-end pipeline."""
    try:
        from core.koharu_pipeline import KoharuPipeline
    except ImportError as e:
        logger.error("[Koharu] Initialization failed: %s", e)
        return []

    pipeline = KoharuPipeline(cfg=cfg)
    all_bubble_results = []
    seen_bubbles = []  # for dedup
    from core.translator import Translator, BubbleResult

    # Initialize the translator instance once for the page/batch
    series = cfg.get("series", "Unknown") 
    translator = Translator(series, cfg=cfg)

    for page_idx, img_path in enumerate(images):
        page_num = page_idx + 1
        logger.info("  [Koharu] Page %d/%d: %s", page_num, total_pages, Path(img_path).name)

        try:
            # 1. Detect, 2. OCR, 3. Font Analysis, 4. Inpaint
            result_image, raw_bubbles = pipeline.process_page(img_path, page_num)

            # Save the inpainted image to bg_cache for instant re-rendering
            bg_cache_path = os.path.join(bg_cache_dir, f"page_{page_num:04d}.png")
            result_image.save(bg_cache_path, format="PNG")

            from core.translator import _get_font_cfg
            font_cfg = _get_font_cfg(cfg, series)
            render_regions = []

            # 5. Translate & 6. Setup for Rendering
            for b_idx, mb in enumerate(raw_bubbles):
                if not mb.source_text.strip():
                    continue

                # Chunk deduplication
                if chunk_meta and _is_duplicate_bubble(
                    mb.y, mb.source_text, page_idx, chunk_meta, seen_bubbles,
                ):
                    logger.debug("    [Koharu] DEDUP skip: %r", mb.source_text[:40])
                    continue
                
                # Setup translation
                engine = cfg.get("translation_engine", "nllb")
                
                if engine != "manual":
                    result = translator.translate_text(mb.source_text)
                    mb.translated_text = result.translated_text
                else:
                    mb.translated_text = mb.source_text
                    result = BubbleResult(
                        source_text=mb.source_text,
                        translated_text="",
                        source_lang="",
                        confidence=0.0,
                        source="manual",
                        approved=False,
                        edited=False,
                        memory_id=None
                    )

                # Record for future dedup
                if chunk_meta:
                    global_y = _bubble_to_global_y(mb.y, page_idx, chunk_meta)
                    seen_bubbles.append((global_y, mb.source_text))

                # Save crop
                original_image = Image.open(img_path).convert("RGB")
                crop_url = ""
                try:
                    crop = original_image.crop((mb.x, mb.y, mb.x + mb.w, mb.y + mb.h))
                    crop_name = f"p{page_num:04d}_b{b_idx:04d}.jpg"
                    crop.save(os.path.join(crops_dir, crop_name), format="JPEG", quality=85)
                    crop_url = f"/session_crop/{cbz_name.replace('.cbz', '')}/{crop_name}"
                except Exception as crop_exc:
                    logger.debug("Crop save failed: %s", crop_exc)

                # Store for review/DB
                all_bubble_results.append((result, mb, page_num, crop_url))

                from core.inpainter import BubbleRegion
                br = BubbleRegion(
                    x=mb.x, y=mb.y, w=mb.w, h=mb.h,
                    source_text=mb.source_text,
                    translated_text=mb.translated_text,
                    font_cfg=font_cfg
                )
                render_regions.append(br)

            # 7. Render text onto the final image
            from core.inpainter import Inpainter
            inpainter_instance = Inpainter(cfg=cfg)
            out_page = os.path.join(output_tmp, Path(img_path).stem + ".png")
            final = inpainter_instance.render_text(result_image, render_regions)
            final.save(out_page, format="PNG", optimize=False)

            if progress_callback:
                progress_callback(page_idx + 1, total_pages)

        except Exception as exc:
            logger.error("  [Koharu] Error processing %s: %s", Path(img_path).name, exc, exc_info=True)
            shutil.copy2(img_path, os.path.join(output_tmp, Path(img_path).name))
            if progress_callback:
                progress_callback(page_idx + 1, total_pages)

    return all_bubble_results

def _process_pages_mit(
    images, output_tmp, crops_dir, cbz_name,
    cfg, logger, progress_callback, total_pages,
    chunk_meta=None,
):
    """Process all pages using MIT's full end-to-end pipeline."""
    from core.mit_pipeline import MITPipeline, MITBubbleResult
    from core.translator import BubbleResult

    pipeline = MITPipeline(cfg=cfg)
    all_bubble_results = []
    seen_bubbles = []  # for dedup

    for page_idx, img_path in enumerate(images):
        page_num = page_idx + 1
        logger.info("  [MIT] Page %d/%d: %s", page_num, total_pages, Path(img_path).name)

        try:
            result_image, bubbles = pipeline.translate_page(img_path)

            if result_image is None:
                out_page = os.path.join(output_tmp, Path(img_path).name)
                shutil.copy2(img_path, out_page)
                logger.warning("    MIT returned no result, copied original.")
                if progress_callback:
                    progress_callback(page_idx + 1, total_pages)
                continue

            # Save the result image
            out_page = os.path.join(output_tmp, Path(img_path).stem + ".png")
            result_image.save(out_page, format="PNG")

            # Convert MIT bubbles → BubbleResult for session saving
            original_image = Image.open(img_path).convert("RGB")
            for b_idx, mb in enumerate(bubbles):
                if not mb.source_text.strip():
                    continue

                # Chunk deduplication
                if chunk_meta and _is_duplicate_bubble(
                    mb.y, mb.source_text, page_idx, chunk_meta, seen_bubbles,
                ):
                    logger.debug("    [MIT] DEDUP skip: %r", mb.source_text[:40])
                    continue

                # Save crop from original image
                crop_url = ""
                try:
                    bbox = (mb.x, mb.y, mb.x + mb.w, mb.y + mb.h)
                    crop = original_image.crop(bbox)
                    crop_name = f"p{page_num:04d}_b{b_idx:04d}.jpg"
                    crop.save(os.path.join(crops_dir, crop_name),
                              format="JPEG", quality=85)
                    crop_url = f"/session_crop/{cbz_name.replace('.cbz', '')}/{crop_name}"
                except Exception as crop_exc:
                    logger.debug("Crop save failed: %s", crop_exc)

                br = BubbleResult(
                    source_text=mb.source_text,
                    translated_text=mb.translated_text,
                    source_lang="",
                    confidence=90.0,
                    source="mit_pipeline",
                    approved=False,
                    edited=False,
                    memory_id=None,
                )
                all_bubble_results.append((br, mb, page_num, crop_url))

                # Record for future dedup
                if chunk_meta:
                    global_y = _bubble_to_global_y(mb.y, page_idx, chunk_meta)
                    seen_bubbles.append((global_y, mb.source_text))

                logger.debug("    [MIT] Bubble: %r → %r",
                             mb.source_text[:30],
                             mb.translated_text[:30])

        except Exception as exc:
            logger.error("    [MIT] Page %d error: %s", page_num, exc, exc_info=True)
            out_page = os.path.join(output_tmp, Path(img_path).name)
            shutil.copy2(img_path, out_page)

        if progress_callback:
            progress_callback(page_idx + 1, total_pages)

    pipeline.unload()
    return all_bubble_results


def _save_session_data(cbz_name: str, bubble_results, series: str, chunk_meta: dict = None):
    """
    Save bubble translation results to a JSON session file for the review UI.
    bubble_results is a list of (BubbleResult, page_num, crop_url) tuples.
    Stored at: web/sessions/<cbz_name>.json
    """
    sessions_dir = os.path.join(_ROOT, "web", "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    out_path = os.path.join(sessions_dir, cbz_name.replace(".cbz", "") + ".json")

    data = {
        "cbz_name": cbz_name,
        "series":   series,
        "created":  datetime.now().isoformat(),
        "chunk_meta": chunk_meta,
        "bubbles":  [
            {
                "source_text":     b.source_text,
                "translated_text": b.translated_text,
                "source_lang":     b.source_lang,
                "confidence":      b.confidence,
                "source":          b.source,
                "approved":        b.approved,
                "edited":          b.edited,
                "memory_id":       b.memory_id,
                "page_num":        page_num,
                "crop_url":        crop_url,
                "x":               region.x,
                "y":               region.y,
                "w":               region.w,
                "h":               region.h,
            }
            for b, region, page_num, crop_url in bubble_results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_batch(input_dir: str, output_dir: str, series: str, cfg: dict,
              logger: logging.Logger):
    """Process all CBZ files in input_dir."""
    cbz_files = sorted(Path(input_dir).glob("*.cbz"))
    if not cbz_files:
        logger.warning("No CBZ files found in: %s", input_dir)
        return

    logger.info("Found %d CBZ files in %s", len(cbz_files), input_dir)
    batch_start = time.time()
    total_bubbles = 0
    successes = 0

    for i, cbz_path in enumerate(cbz_files, 1):
        logger.info("\n[%d/%d] %s", i, len(cbz_files), cbz_path.name)
        result = process_cbz(str(cbz_path), output_dir, series, cfg, logger)

        if result.get("skipped"):
            continue
        if result["success"]:
            successes    += 1
            total_bubbles += result["num_bubbles"]
        else:
            logger.error("Failed: %s — %s", cbz_path.name, result.get("error"))

    batch_elapsed = time.time() - batch_start

    # ── Trigger fine-tuning if enough approved pairs ──────────────────────────
    mm         = MemoryManager()
    approved   = mm.count_approved_all_series()
    min_pairs  = cfg.get("memory", {}).get("min_pairs_to_finetune", 10)

    if approved >= min_pairs:
        logger.info("Approved pairs: %d ≥ %d. Triggering fine-tune...", approved, min_pairs)
        try:
            from model.trainer import Trainer
            trainer = Trainer(cfg)
            version = trainer.finetune()
            notify_finetune_done(version, approved)
        except Exception as exc:
            logger.error("Fine-tuning error: %s", exc, exc_info=True)
    else:
        logger.info("Approved pairs: %d < %d. Skipping fine-tune.", approved, min_pairs)

    # ── Toast notification ────────────────────────────────────────────────────
    notify_batch_done(
        cbz_name  = f"{successes}/{len(cbz_files)} files",
        num_bubbles=total_bubbles,
        elapsed_secs=batch_elapsed,
    )

    logger.info(
        "\n✓ Batch done: %d/%d succeeded, %d bubbles translated, %.1fs total.",
        successes, len(cbz_files), total_bubbles, batch_elapsed,
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

def _save_crop(image, region, page_num, b_idx, crops_dir, cbz_name) -> str:
    """Helper to save a speech bubble crop for the review UI."""
    try:
        crop = image.crop(region.bbox)
        crop_name = f"p{page_num:04d}_b{b_idx:04d}.jpg"
        crop.save(os.path.join(crops_dir, crop_name), format="JPEG", quality=85)
        return f"/session_crop/{cbz_name.replace('.cbz', '')}/{crop_name}"
    except Exception as e:
        logging.getLogger("batch_processor").debug("Crop save failed: %s", e)
        return ""

def main():
    parser = argparse.ArgumentParser(
        description="CBZ Translator — Batch Processor",
    )
    parser.add_argument("--input",  default="./input",  help="Folder with CBZ files")
    parser.add_argument("--output", default=None,       help="Output folder (overrides settings.yaml)")
    parser.add_argument("--series", default="Unknown",  help="Series name (for memory & font settings)")
    parser.add_argument("--config", default=None,       help="Path to settings.yaml")
    args = parser.parse_args()

    cfg    = _load_cfg(args.config)
    logger = _setup_logging(os.path.join(_ROOT, "logs"))

    output_dir = (
        args.output
        or cfg.get("output_folder", "./output")
    )
    # Resolve relative paths from project root
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)
    input_dir = args.input
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(_ROOT, input_dir)

    os.makedirs(input_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    run_batch(input_dir, output_dir, args.series, cfg, logger)


if __name__ == "__main__":
    main()
