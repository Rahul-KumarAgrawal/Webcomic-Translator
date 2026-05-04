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
1.  **Pororo Elite (🥇 New)**: High-speed ONNX-accelerated engine for Korean/Japanese. Includes "Global Compatibility Shield" for stability.
2.  **Manga-OCR (Default)**: The standard for high-accuracy Japanese text.
3.  **PPOCR-v5**: Modern PaddleOCR v4/v5 architecture (Ogkalu PyTorch port included).
4.  **Tesseract-OCR**: Reliable local fallback (Requires install into `./Tesseract-OCR/`).
5.  **MIT Mayo (JPN)**: Ultra-fast 48px OCR for rapid processing.

## 📦 Setup
1.  Run `setup.bat` to install all dependencies and verify model paths.
2.  Run `.\python\python.exe scripts/download_new_models.py` to ensure you have the latest elite weights.
3.  Start the UI: `python app.py`

## 🖥️ Hardware Requirements
*   **GPU**: NVIDIA RTX Series (3050+) recommended.
*   **VRAM**: 4GB+ for basic, 8GB+ for elite engines.

## 🌟 Recent Updates (May 2026)
*   **Pororo Elite Upgrade**: Fully refactored the Korean OCR stack. Now uses ONNX-accelerated BrainOCR + CRAFT with optimized 64px input dimensions.
*   **Global Compatibility Shield**: Implemented a runtime patching system that automatically "heals" library conflicts between modern Python/Pillow/NumPy and legacy AI engines.
*   **Modular Routing**: Stabilized the "Ghost Code" in `inpainter.py`, ensuring correct engine selection for Korean and Japanese.
*   **Tesseract Fallback**: Integrated local Tesseract-OCR support for mission-critical reliability.
