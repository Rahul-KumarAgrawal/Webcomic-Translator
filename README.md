# CBZ Translator

A fully local, offline CBZ manga/manhua/manhwa → English translation tool.
Runs on **Windows** with an **NVIDIA RTX 3050 (4 GB VRAM)**.

---

## Features

| Feature | Details |
|---|---|
| **Input**  | CBZ files (drag-and-drop) or folders |
| **Output** | Clean CBZ with translated text cleanly rendered into bubbles |
| **OCR Engines** | MangaOCR, PaddleOCR, EasyOCR, MIT (Modular Pipeline) |
| **Detection** | YOLO (Pipeline Koharu), ComicTextSegmenter, MIT |
| **Inpainting** | LaMa, PanelCleaner (Modular Background Cleaning) |
| **Webtoon Support** | Advanced vertical stitching and coordinate-aware overlap chunking |
| **Translation** | Offline (NLLB-200 600M) + APIs (DeepL, Google Gemini, OpenAI, Groq, Baidu) |
| **Chapter Batching** | 3-Phase Pipeline: OCR → Batch Translation → Chapter Rendering |
| **Memory** | Per-series + global SQLite/JSON translation memory |
| **Learning** | LoRA fine-tuning on user-approved pairs (PEFT) |
| **Review UI** | Local Flask web app at `localhost:5000` |

---

## Quick Start

### 1. Prerequisites
- Python 3.10+ in PATH
- NVIDIA GPU drivers with CUDA 11.8+
- Git (optional, for cloning)

### 2. NVIDIA GPU & DLL Support
The project now includes an automated **Super-Linker** to handle common Windows DLL issues (like `cudnn64_8.dll` not found).
- **Automated Dependency Management:** `setup.bat` installs all required NVIDIA binaries directly into the local environment.
- **Dynamic Linking:** The application automatically discovers and links `cuDNN`, `cuBLAS`, and other CUDA components at startup.
- **Hardware Optimization:** PaddleOCR 2.8.1 is pre-configured for RTX GPUs with stable 2.x API calls.
- **Drive Redirection:** All AI models are automatically redirected to the project's `model/` folder on the D: drive to save C: drive space.

### 2. First-time Setup
Double-click **`setup.bat`** (or run in PowerShell):
```
setup.bat
```
This will:
- Create all directories
- Install PyTorch with CUDA
- Download NLLB-200 600M to `model/base/`
- Initialise global translation memory DB

### 3. Launch Web UI
```
run_web.bat
```
Open http://localhost:5000 in your browser.

### 4. Batch CLI (no UI)
```
run_batch.bat --input ./input --output ./output --series "Solo Leveling"
```

---

## Project Structure

```
cbz-translator/
├── input/                    # Drop CBZ files here for batch processing
├── output/                   # Translated CBZ files
├── logs/                     # Per-session log files
├── fonts/                    # User-uploaded .ttf fonts
├── backups/                  # Auto-backups (never auto-deleted)
├── model/                    # AI models (NLLB, YOLO, OCR)
├── config/settings.yaml      # All user configuration
└── ...

---

## Web UI Pages

| URL | Purpose |
|---|---|
| `/` | Upload CBZ files, monitor translation queue |
| `/review/<name>` | Review every bubble — approve / edit / reject |
| `/memory` | Browse and search global + per-series memory |
| `/train` | Training history, manual fine-tune trigger |
| `/backup` | List backups, create manual backup, restore |
| `/settings` | Output path, GPU device, font settings |

---

## Translation Memory

- **Series memory** is checked first (approved pairs only)
- **Global memory** is checked second (approved pairs only)
- If no match: **NLLB-200 inference** → saved as unapproved
- Approving/editing in the review UI → saved to memory + training data
- Confidence score shown in UI (stored in DB only, never in output CBZ)

---

## Fine-Tuning

- Requires ≥ 10 approved pairs
- Uses **LoRA (PEFT)** — only adapter weights saved (~small files)
- Triggered automatically after batch processing, or manually via `/train`
- Checkpoints versioned: `model/finetuned/checkpoint_v1/`, `v2/`, ...
- `model/finetuned/latest/` always points to the best checkpoint
- Auto-backup runs after each checkpoint

---

## Configuration (`config/settings.yaml`)

```yaml
output_folder: ./output
gpu_device: cuda

default_font:
  family: ./fonts/my_font.ttf   # null = bundled default
  color: "#FFFFFF"
  size: auto                     # "auto" or integer

series_fonts:
  Solo Leveling:
    family: ./fonts/custom.ttf
    color: "#000000"
    size: 14
```

---

## VRAM Budget (RTX 3050, 4 GB)

| Component | VRAM usage |
|---|---|
| NLLB-200 600M inference (fp16) | ~1.5 GB |
| LoRA training (r=8) | ~2.8 GB total |
| Pipeline Koharu (YOLO) + MangaOCR | ~1.5 GB |
| PaddleOCR (Multilingual) | ~1.0 GB |
| LaMa / PanelCleaner | ~1.0 - 1.5 GB |
| manga-image-translator (Legacy) | ~1.5 GB |

> Training and inference are never run simultaneously. Models are dynamically loaded and unloaded as needed to prevent VRAM overflow.

---

## Supported Languages

| Language | NLLB code | ISO (Gemini) |
|---|---|---|
| Chinese (Simplified) | `zho_Hans` | `zh-CN` |
| Japanese | `jpn_Jpan` | `ja` |
| Korean | `kor_Hang` | `ko` |
| English | `eng_Latn` | `en` |

---

## License

Personal use only. See individual library licenses:
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator)
- [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0)
- [PEFT](https://github.com/huggingface/peft) (Apache 2.0)
