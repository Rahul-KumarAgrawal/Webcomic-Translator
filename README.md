# Manga Translation Pipeline (Koharu Version)

This project is a high-performance, modular manga translation pipeline designed for NVIDIA GPUs (RTX Series).

## 🚀 Key Features
*   **Modular Detection**: Switch between high-speed YOLOv8 and high-fidelity Mayo/Ogkalu engines.
*   **Dual-Pass OCR**: Combines fast MIT OCR for filtering and high-quality Manga-OCR for translation.
*   **Hardware Accelerated**: Full CUDA support for all detection and OCR models.
*   **Smart Gap-Filling**: Uses PaddleOCR to find text areas missed by standard detectors.

## 🛠️ Engine Lineup

### Detection Engines
1.  **Mayo Github (🥇 Best)**: The classic gold standard for high-fidelity bubble shapes. Now optimized via the MIT CTD engine.
2.  **Ogkalu Stable (Dual)**: Extremely robust YOLOv8m models for both text and bubbles.
3.  **Ogkalu Combine**: Single-pass YOLOv8m. Fastest high-quality engine.
4.  **YOLO Only**: Pure text detection for speed.

### OCR Engines
1.  **Manga-OCR (Default)**: The standard for high-accuracy Japanese text.
2.  **MIT Mayo (JPN)**: Ultra-fast 48px OCR for rapid processing.
3.  **PPOCR-v5**: Modern PaddleOCR v4/v5 architecture.
4.  **Pororo**: Specialized Korean/Japanese OCR engine.

## 📦 Setup
1.  Run `setup.bat` to install all dependencies and verify model paths.
2.  Run `.\python\python.exe scripts/download_new_models.py` to ensure you have the latest elite weights.
3.  Start the UI: `python app.py`

## 🖥️ Hardware Requirements
*   **GPU**: NVIDIA RTX Series (3050+) recommended.
*   **VRAM**: 4GB+ for basic, 8GB+ for elite engines.
