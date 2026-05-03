"""
model/model_loader.py
Loads the translation model (NLLB-200 600M).

Priority:
  1. model/finetuned/latest/  (LoRA adapter merged onto base)
  2. model/base/              (vanilla NLLB-200 600M)

Handles 4-bit quantisation fallback if VRAM is insufficient.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelLoader:
    """
    Loads and returns (model, tokenizer) for NLLB-200 translation.
    """

    BASE_PATH      = os.path.join(_ROOT, "model", "base")
    FINETUNED_PATH = os.path.join(_ROOT, "model", "finetuned", "latest")

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = cfg.get("gpu_device", "cuda")

    def load(self):
        """
        Load and return (model, tokenizer).
        Tries finetuned/latest first, then falls back to base model.
        """
        if Path(self.FINETUNED_PATH).exists():
            logger.info("Loading fine-tuned model from: %s", self.FINETUNED_PATH)
            try:
                return self._load_finetuned(self.FINETUNED_PATH)
            except Exception as exc:
                logger.warning("Failed to load fine-tuned model (%s), falling back to base.", exc)

        logger.info("Loading base NLLB-200 model from: %s", self.BASE_PATH)
        return self._load_base(self.BASE_PATH)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_base(self, model_path: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            logger.info("NLLB-200 model not found locally. Downloading on demand (this may take a while)...")
            from huggingface_hub import snapshot_download
            os.makedirs(model_path, exist_ok=True)
            snapshot_download(
                repo_id="facebook/nllb-200-distilled-600M",
                local_dir=model_path,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # Try full precision first; fall back to 4-bit if VRAM is insufficient
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            logger.info("Model loaded in fp16 on %s.", self.device)
        except RuntimeError as exc:
            logger.warning("fp16 load failed (%s), trying 4-bit quantisation...", exc)
            model = self._load_4bit(model_path)

        return model, tokenizer

    def _load_finetuned(self, adapter_path: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_PATH, local_files_only=True
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.BASE_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()   # merge LoRA weights for faster inference
        model = model.to(self.device)
        logger.info("Fine-tuned model (LoRA merged) loaded on %s.", self.device)
        return model, tokenizer

    def _load_4bit(self, model_path: str):
        from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
        import torch

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        logger.info("Model loaded in 4-bit quantisation (VRAM fallback).")
        return model
