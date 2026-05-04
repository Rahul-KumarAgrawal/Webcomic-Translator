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
*   **Symbol-Only Fallback**: Integrated a specialized Tesseract pass to automatically recover punctuation-only bubbles (e.g., `...`, `!!`) missed by standard OCR.
*   **Dynamic Language Mapping**: All OCR engines (Tesseract, EasyOCR, Paddle) now dynamically map NLLB codes (e.g., `kor_Hang`) to their internal engine codes.

## 💡 Elite Translation Tips

### 1. High-Precision Language Selection
For maximum OCR accuracy, use the full NLLB-200 language codes in the UI:
*   **Korean**: `kor_Hang` (Enables the Elite Pororo stack)
*   **Japanese**: `jpn_Jpan` (Enables high-fidelity Manga-OCR)
*   **Chinese (Simplified)**: `zho_Hans`
*   **English**: `eng_Latn`

### 2. Tesseract Symbol Recovery
If bubbles containing only `...` or `!!!` are coming back empty, ensure you have Tesseract installed in the `./Tesseract-OCR/` folder. The pipeline will automatically use it as a "Smart De-Fragmenter" and symbol-recovery engine.

### 3. Fixing "Fragmented" Text
If Tesseract returns words with spaces (e.g., `인 정 했`), the pipeline now includes a **Smart De-Fragmenter** that automatically collapses these spaces while preserving English word breaks.
