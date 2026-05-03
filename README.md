# CBZ Translator

A fully local, offline CBZ manga/manhua/manhwa → English translation tool.
Runs on **Windows** with an **NVIDIA RTX 3050 (4 GB VRAM)**.

---

## Features

| Feature | Details |
|---|---|
| **Input**  | CBZ files (drag-and-drop) or folders |
| **Output** | Clean CBZ with translated text cleanly rendered into bubbles |
| **Detection Engines** | YOLO Hybrid, Koharu Dual, YSG v1 (ogkalu), YSG v2 (Kitsumed), MIT |
| **Inpainting** | LaMa, PanelCleaner, **Solid Fill (Perfect for Manhwa)** |
| **Webtoon Support** | Advanced vertical stitching and coordinate-aware overlap chunking |
| **Translation** | Offline (NLLB-200 600M) + APIs (DeepL, Google Gemini, OpenAI, Groq, Baidu) |
| **Batch Pipeline** | **3-Phase (OCR → Chapter Translation → Final Render)** |
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
| Waifu2x (2x Page Upscale) | ~1.0 - 2.0 GB (Peak) |
| YOLO Hybrid / Koharu Dual | ~1.5 GB |
| YSG v1 / v2 (ogkalu/Kitsumed) | ~0.8 GB |
| MangaOCR (Expert Gap-Filling) | ~1.2 GB |
| LaMa / PanelCleaner / Solid Fill | ~1.0 - 1.5 GB |
| manga-image-translator (Legacy) | ~1.5 GB |

> Training and inference are never run simultaneously. Models are dynamically loaded and unloaded as needed to prevent VRAM overflow.

---

## Supported Languages

| Language | NLLB code | ISO (Gemini) | Detection |
|---|---|---|---|
| Chinese (Simplified) | `zho_Hans` | `zh-CN` | Auto / Manual |
| Japanese | `jpn_Jpan` | `ja` | Auto / Manual |
| Korean | `kor_Hang` | `ko` | Auto / Manual |
| English | `eng_Latn` | `en` | Auto / Manual |

---

## Language Auto-Detection

The pipeline automatically identifies the source language using a tiered approach:
1. **Google Gemini API** (Highest accuracy, requires API key)
2. **Google Cloud Vision** (Standard cloud OCR)
3. **Local EasyOCR + langdetect** (Fully offline fallback)

---

## License

Personal use only. See individual library licenses:
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator)
- [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0)
- [PEFT](https://github.com/huggingface/peft) (Apache 2.0)
