"""
core/mit_pipeline.py
Wrapper around manga-image-translator's full end-to-end pipeline.

When the user toggles "Use MIT Full Pipeline", this module handles
the entire flow: detect → OCR → translate → inpaint → render
via MIT's MangaTranslator.translate() in one shot.

MIT handles all rendering internally using its own renderer
(manga2eng, default, etc.) configured via settings.yaml → mit.renderer.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Force model downloads to D drive instead of wherever the package is installed
cache_dir = os.path.join(_ROOT, "model", "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
os.environ["MIT_MODELS_DIR"] = os.path.join(_ROOT, "models")
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir

# Paddle caches (PaddleOCR / PaddlePaddle / PaddleX)
paddle_cache = os.path.join(_ROOT, "model", "paddle_cache")
os.makedirs(paddle_cache, exist_ok=True)
os.environ["PPOCR_HOME"]   = paddle_cache
os.environ["PADDLE_HOME"]  = paddle_cache
os.environ["PADDLEX_HOME"] = paddle_cache
os.environ["HUB_HOME"]     = paddle_cache


@dataclass
class MITBubbleResult:
    """A text region extracted from MIT's translation context."""
    source_text: str
    translated_text: str
    x: int
    y: int
    w: int
    h: int


class MITPipeline:
    """
    Wraps MIT's MangaTranslator to run the full pipeline on a single page.

    MIT handles detect → OCR → translate → inpaint → render internally.
    The renderer used is controlled by settings.yaml → mit.renderer.

    Usage:
        pipeline = MITPipeline(cfg)
        result_image, bubbles = pipeline.translate_page("page.png")
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._mit_cfg = cfg.get("mit", {})
        self._device = cfg.get("gpu_device", "cuda")
        self._translator_obj = None

        # Set environment variables for MIT translators that need API keys
        self._set_api_keys()

    def _set_api_keys(self):
        """Set environment variables for MIT's cloud translators."""
        mit_keys = self.cfg.get("mit_api_keys", {})
        env_map = {
            "openai_api_key": "OPENAI_API_KEY",
            "gemini_api_key": "GOOGLE_GEMINI_API_KEY",
            "groq_api_key": "GROQ_API_KEY",
            "deepl_api_key_mit": "DEEPL_AUTH_KEY",
        }
        for cfg_key, env_var in env_map.items():
            val = mit_keys.get(cfg_key, "")
            if val:
                os.environ[env_var] = val

    def _ensure_translator(self):
        """Lazy-initialize the MangaTranslator object."""
        if self._translator_obj is not None:
            return

        from manga_translator import MangaTranslator

        # Resolve font path for MIT's rendering engine
        font_path = self._mit_cfg.get("font_path", None)
        if font_path is None:
            for candidate in [
                os.path.join(_ROOT, "fonts", "NotoSansMonoCJK-VF.ttf.ttc"),
                os.path.join(_ROOT, "fonts", "comic shanns 2.ttf"),
                "C:/Windows/Fonts/Arial.ttf",
                "C:/Windows/Fonts/Calibri.ttf",
            ]:
                if os.path.exists(candidate):
                    font_path = candidate
                    break

        params = {
            "use_gpu": self._device.startswith("cuda"),
            "verbose": False,
            "kernel_size": 3,
            "pre_dict": None,
            "post_dict": None,
            "font_path": font_path,
        }
        self._translator_obj = MangaTranslator(params)

    def _build_config(self):
        """Build MIT Config object from our settings.yaml."""
        from manga_translator.config import (
            Config,
            Detector as MITDetector,
            Inpainter as MITInpainter,
            Ocr as MITOcr,
            Renderer as MITRenderer,
            Translator as MITTranslator,
        )

        cfg = Config()

        # Detector
        det_key = self._mit_cfg.get("detector", "default")
        try:
            cfg.detector.detector = MITDetector(det_key)
        except (ValueError, KeyError):
            cfg.detector.detector = MITDetector.default

        # OCR
        ocr_key = self._mit_cfg.get("ocr", "48px")
        try:
            cfg.ocr.ocr = MITOcr(ocr_key)
        except (ValueError, KeyError):
            cfg.ocr.ocr = MITOcr.ocr48px

        # Inpainter
        inp_key = self._mit_cfg.get("inpainter", "default")
        try:
            cfg.inpainter.inpainter = MITInpainter(inp_key)
        except (ValueError, KeyError):
            cfg.inpainter.inpainter = MITInpainter.default

        # Translator
        trans_key = self._mit_cfg.get("translator", "sugoi")
        try:
            cfg.translator.translator = MITTranslator(trans_key)
        except (ValueError, KeyError):
            cfg.translator.translator = MITTranslator.sugoi

        # Target language (MIT uses uppercase codes like ENG, JPN, CHS, KOR)
        cfg.translator.target_lang = self._mit_cfg.get("target_lang", "ENG")

        # ── Renderer: use MIT's own renderer ────────────────────────────────
        # Configured via settings.yaml → mit.renderer (default, manga2eng, etc.)
        rend_key = self._mit_cfg.get("renderer", "default")
        try:
            cfg.render.renderer = MITRenderer(rend_key)
        except (ValueError, KeyError):
            cfg.render.renderer = MITRenderer.default

        # ── Mask dilation ────────────────────────────────────────────────────
        cfg.mask_dilation_offset = self._mit_cfg.get("mask_dilation_offset", 0)

        return cfg

    def translate_page(
        self, image_path: str
    ) -> Tuple[Optional[Image.Image], List[MITBubbleResult]]:
        """
        Run MIT's full pipeline on a single page image.

        Returns
        -------
        (result_image, bubbles)
            result_image : PIL Image with translated text rendered by MIT, or None on error
            bubbles      : list of MITBubbleResult with source/translated text + coords
        """
        try:
            self._ensure_translator()
            mit_config = self._build_config()

            image = Image.open(image_path).convert("RGB")

            # MIT's translate() is async — bridge with asyncio
            ctx = self._run_async_translate(image, mit_config)

            # ── Get MIT's rendered result image ──────────────────────────────
            # MIT stores the fully rendered output in ctx.img_translated
            # (numpy array, RGB). Fall back to img_inpainted if rendering
            # was skipped (e.g. renderer=none).
            result_np = getattr(ctx, "img_translated", None)
            if result_np is None:
                result_np = getattr(ctx, "img_inpainted", None)

            if result_np is not None:
                result_image = (
                    Image.fromarray(result_np)
                    if not isinstance(result_np, Image.Image)
                    else result_np
                )
            else:
                logger.warning("MIT pipeline returned no output image, using original.")
                result_image = image.copy()

            # Extract text regions for session/review data
            bubbles = self._extract_bubbles(ctx)

            logger.info(
                "MIT pipeline: %s → %d text regions translated",
                Path(image_path).name,
                len(bubbles),
            )
            return result_image, bubbles

        except Exception as exc:
            import traceback
            logger.error(
                "MIT pipeline error for %s: %s\n%s",
                image_path, exc, traceback.format_exc(),
            )
            return None, []

    def _run_async_translate(self, image, mit_config):
        """Run MIT's async translate() method synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(self._sync_translate_in_new_loop, image, mit_config)
                    return future.result(timeout=600)  # 10 min timeout
            else:
                return loop.run_until_complete(
                    self._translator_obj.translate(image, mit_config)
                )
        except RuntimeError:
            return self._sync_translate_in_new_loop(image, mit_config)

    def _sync_translate_in_new_loop(self, image, mit_config):
        """Create a fresh event loop and run translate()."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._translator_obj.translate(image, mit_config)
            )
        finally:
            loop.close()

    def _extract_bubbles(self, ctx) -> List[MITBubbleResult]:
        """Extract text region info from MIT's Context for session saving."""
        bubbles = []
        text_regions = getattr(ctx, "text_regions", None) or []

        for tr in text_regions:
            xyxy = getattr(tr, "xyxy", None)
            if xyxy is None:
                aabb = getattr(tr, "aabb", None)
                if aabb is not None:
                    xyxy = aabb
            if xyxy is None:
                continue

            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            source = getattr(tr, "text", "") or ""
            translated = getattr(tr, "translation", "") or ""

            bubbles.append(MITBubbleResult(
                source_text=source,
                translated_text=translated,
                x=x1, y=y1,
                w=x2 - x1, h=y2 - y1,
            ))

        return bubbles

    def unload(self):
        """Release GPU memory."""
        self._translator_obj = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
