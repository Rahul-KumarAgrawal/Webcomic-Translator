# CBZ Translator - Knowledge Graph Report

## 🌐 Architectural Overview
This project is a sophisticated **offline manga translation system** optimized for consumer-grade hardware (specifically targetting **4GB VRAM**). It leverages a multi-pipeline architecture to handle diverse comic styles and layout complexities.

### 🧩 Core Components
*   **Translation Engine (`core/translator.py`)**: A multi-stage pipeline that prioritizes existing translation memory before falling back to NLLB-200 or cloud APIs.
*   **Model Pipeline (`mit_pipeline.py`, `koharu_pipeline.py`)**: Wrappers for advanced OCR and detection frameworks.
*   **Memory Manager (`memory/memory_manager.py`)**: SQLite-backed system for persistence of translation pairs.
*   **LoRA Trainer (`model/trainer.py`)**: Implements on-device learning by fine-tuning model adapters based on user corrections.

## 🚀 Key Workflows
1.  **Translation Pipeline**:
    *   `Series Memory` → `Global Memory` → `NLLB-200 Model` (or API fallback)
2.  **Learning Loop**:
    *   `Review UI` (Correction) → `Approved Pair` (DB) → `LoRA Trainer` (Checkpoint)

## ⚖️ VRAM Strategy
The project implements a strict **VRAM Budgeting** strategy for an RTX 3050:
*   **Inference**: ~1.5 GB (fp16)
*   **Fine-tuning**: ~2.8 GB
*   **Strategy**: Model unloading and 4-bit quantization are used to stay within the 4GB limit.

## 🔗 Surprising Connections
*   **MIT ↔ Koharu**: The system includes two separate end-to-end pipelines, allowing users to switch based on whether the manga is Japanese (MIT) or has complex layouts (Koharu).
*   **SSE Streaming**: The web app uses Server-Sent Events (`progress_stream`) for real-time progress updates, a modern pattern for long-running AI tasks.
