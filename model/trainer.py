"""
model/trainer.py
LoRA fine-tuning of NLLB-200 600M on all approved translation pairs.

Rules:
  - Only runs if ≥ 10 approved pairs exist (enforced by caller)
  - Saves adapter checkpoint to model/finetuned/checkpoint_vN/
  - Updates model/finetuned/latest/ to the new checkpoint
  - Appends new pairs to model/training_data/approved_pairs.jsonl
  - Triggers backup and toast notification after each checkpoint
  - Logs training history to model/training_data/training_log.json
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINETUNED_DIR     = os.path.join(_ROOT, "model", "finetuned")
BASE_PATH         = os.path.join(_ROOT, "model", "base")
TRAINING_DATA_DIR = os.path.join(_ROOT, "model", "training_data")
APPROVED_JSONL    = os.path.join(TRAINING_DATA_DIR, "approved_pairs.jsonl")
REJECTED_JSONL    = os.path.join(TRAINING_DATA_DIR, "rejected_pairs.jsonl")
TRAINING_LOG      = os.path.join(TRAINING_DATA_DIR, "training_log.json")


class Trainer:
    """Fine-tunes NLLB-200 with LoRA on approved translation pairs."""

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = cfg.get("gpu_device", "cuda")
        self.train_cfg = cfg.get("training", {})

    def finetune(self, progress_cb=None) -> int:
        """
        Run a fine-tuning pass. Returns the checkpoint version number.
        """
        start_time = time.time()
        logger.info("Starting LoRA fine-tuning...")

        # 1. Load approved pairs
        pairs = self._load_approved_pairs()
        if not pairs:
            logger.warning("No approved pairs found. Skipping fine-tune.")
            return 0
        logger.info("Training on %d approved pairs.", len(pairs))

        # 2. Set up checkpoint directory
        version    = self._next_version()
        ckpt_dir   = os.path.join(FINETUNED_DIR, f"checkpoint_v{version}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 3. Build HuggingFace Dataset
        from datasets import Dataset
        dataset   = Dataset.from_list(pairs)
        train_set = dataset  # All pairs used for training (small dataset)

        # 4. Load base model + tokenizer
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, TaskType, get_peft_model

        logger.info("Loading base model for training...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
        model     = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,
        )

        # 5. Apply LoRA
        lora_cfg = LoraConfig(
            task_type     = TaskType.SEQ_2_SEQ_LM,
            r             = int(self.train_cfg.get("lora_r", 8)),
            lora_alpha    = int(self.train_cfg.get("lora_alpha", 16)),
            lora_dropout  = float(self.train_cfg.get("lora_dropout", 0.1)),
            target_modules= self.train_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias          = "none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # 6. Tokenise dataset
        max_len = int(self.train_cfg.get("max_seq_length", 256))

        tokenized_ds = train_set.map(
            lambda batch: self._tokenize_batch(tokenizer, batch, max_len),
            batched=True,
            batch_size=32,
        )
        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # 7. Training arguments (VRAM-safe)
        training_args = Seq2SeqTrainingArguments(
            output_dir              = ckpt_dir,
            num_train_epochs        = int(self.train_cfg.get("epochs", 3)),
            per_device_train_batch_size = int(self.train_cfg.get("batch_size", 4)),
            learning_rate           = float(self.train_cfg.get("learning_rate", 2e-4)),
            fp16                    = True,
            gradient_checkpointing  = True,
            save_steps              = 500,
            logging_steps           = 10,
            report_to               = "none",
            predict_with_generate   = False,
            dataloader_num_workers  = 0,    # avoid multiprocessing issues on Windows
        )

        collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        # Build callback if progress_cb provided
        callbacks = []
        if progress_cb:
            from transformers import TrainerCallback
            class _ProgressCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    loss = 0.0
                    if state.log_history:
                        for entry in reversed(state.log_history):
                            if "loss" in entry:
                                loss = entry["loss"]
                                break
                    progress_cb(state.global_step, state.max_steps, loss)
            callbacks.append(_ProgressCallback())

        trainer_obj = Seq2SeqTrainer(
            model         = model,
            args          = training_args,
            train_dataset = tokenized_ds,
            data_collator = collator,
            callbacks     = callbacks,
        )

        logger.info("Training...")
        train_result = trainer_obj.train()
        final_loss   = train_result.training_loss if hasattr(train_result, "training_loss") else 0.0

        # 8. Save LoRA adapter only
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info("Checkpoint v%d saved: %s", version, ckpt_dir)

        # 9. Update latest pointer
        latest_dir = os.path.join(FINETUNED_DIR, "latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(ckpt_dir, latest_dir)
        logger.info("model/finetuned/latest/ updated to v%d", version)

        # 10. Log training history
        elapsed = time.time() - start_time
        self._append_training_log(version, len(pairs), final_loss, elapsed)

        # 11. Unload model from VRAM
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 12. Auto-backup (avoid circular import with web.app)
        try:
            self._run_backup_standalone(version)
        except Exception as exc:
            logger.warning("Auto-backup after fine-tune failed: %s", exc)

        # 13. Toast notification
        from core.notifier import notify_finetune_done
        notify_finetune_done(version, len(pairs))

        return version

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_approved_pairs(self):
        """Read approved pairs from ALL series + global via MemoryManager."""
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from memory.memory_manager import MemoryManager
        mm   = MemoryManager()
        rows = mm.get_all_approved_for_training()

        pairs = [
            {
                "source_lang":    r["source_lang"],
                "source_text":    r["source_text"],
                "translated_text":r["translated_text"],
            }
            for r in rows
        ]
        # NOTE: JSONL is already maintained by the web app's /approve and /edit
        # routes, so we don't re-append here (would cause unbounded duplicates).
        return pairs

    def _tokenize_batch(self, tokenizer, batch, max_len):
        # batch is a dict-of-lists from Dataset.map(batched=True)
        src_texts = [
            f"{lang} {text}"
            for lang, text in zip(batch["source_lang"], batch["source_text"])
        ]
        tgt_texts = batch["translated_text"]
        model_inputs = tokenizer(
            src_texts, max_length=max_len, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=tgt_texts, max_length=max_len, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _run_backup_standalone(self, version: int):
        """Run backup without importing web.app (avoids circular import / Flask startup)."""
        backup_ts  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_dir = os.path.join(_ROOT, "backups", backup_ts)
        os.makedirs(backup_dir, exist_ok=True)

        import shutil as _sh
        # Memory
        memory_src = os.path.join(_ROOT, "memory")
        _sh.copytree(memory_src, os.path.join(backup_dir, "memory"), dirs_exist_ok=True)
        # Latest model checkpoint
        latest_model = os.path.join(_ROOT, "model", "finetuned", "latest")
        if os.path.exists(latest_model):
            _sh.copytree(latest_model, os.path.join(backup_dir, "model_latest"), dirs_exist_ok=True)
        # Training data
        td_src = os.path.join(_ROOT, "model", "training_data")
        if os.path.exists(td_src):
            _sh.copytree(td_src, os.path.join(backup_dir, "training_data"), dirs_exist_ok=True)
        # Manifest
        import json as _json
        with open(os.path.join(backup_dir, "manifest.json"), "w") as f:
            _json.dump({
                "timestamp": backup_ts,
                "triggered_by": f"finetune_v{version}",
                "created_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info("Post-finetune backup created: %s", backup_dir)

    def _next_version(self) -> int:
        """Find the next checkpoint version number."""
        os.makedirs(FINETUNED_DIR, exist_ok=True)
        existing = [
            d.name for d in Path(FINETUNED_DIR).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_v")
        ]
        nums = []
        for name in existing:
            try:
                nums.append(int(name.replace("checkpoint_v", "")))
            except ValueError:
                pass
        return max(nums, default=0) + 1

    def _append_training_log(
        self, version: int, pairs: int, loss: float, elapsed: float
    ) -> None:
        log = []
        if os.path.exists(TRAINING_LOG):
            try:
                with open(TRAINING_LOG, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append({
            "version":   version,
            "pairs":     pairs,
            "loss":      round(loss, 6),
            "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        })
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        with open(TRAINING_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)


def load_training_log():
    """Public helper for the /train route."""
    if not os.path.exists(TRAINING_LOG):
        return []
    try:
        with open(TRAINING_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
