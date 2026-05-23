# Manga Translation Pipeline (Koharu Version)

This project is a high-performance, modular manga translation pipeline designed for NVIDIA GPUs (RTX Series).

## 🚀 Key Features
*   **Decoupled Cloud Architecture**: The user interface is now fully decoupled and deployable to Vercel, allowing access from anywhere.
*   **Permanent ngrok Tunnel**: Secure, zero-config reverse tunneling bridges the Vercel cloud UI directly to your local NVIDIA GPU backend without exposing your IP.
*   **High-Precision Auto-Detect**: New API-driven language detection powered by Groq (LLM). Automatically samples manga pages to identify languages like Japanese, Korean, and accurately distinguishes between Chinese Simplified and Traditional.
*   **Dual-Process Web UI**: Separation of concerns between the main Translation Queue (port 5000) and the Language Detection Service (port 8000) for maximum stability.

## 🛠️ Engine Lineup

### Detection Engines
1.  **Mayo Github (🥇 Best)**: The classic gold standard for high-fidelity bubble shapes. Now optimized via the MIT CTD engine.
2.  **Ogkalu Stable (Dual)**: Extremely robust YOLOv8m models for both text and bubbles.
3.  **Auto-Detect (New)**: Integrated plugin system for language identification.

### OCR Engines
1.  **Pororo Elite (🥇 New)**: High-speed ONNX-accelerated engine for Korean/Japanese. Includes "Global Compatibility Shield" for stability.
2.  **Manga-OCR (Default)**: The standard for high-accuracy Japanese text.
3.  **PPOCR-v5**: Modern PaddleOCR v4/v5 architecture.

## 📦 Setup
1.  Install **Python 3.10+** and **Node.js**.
2.  Run `setup.bat` to install all Python dependencies and download models.
3.  Run `run_web.bat` to launch the full system. This will:
    *   Start the local Flask backend (Port 5000).
    *   Start the Language Detection API (Port 8000).
    *   Open a permanent ngrok tunnel to your backend.
4.  Deploy the `/vercel_deploy` folder to Vercel (`npx vercel --prod`) or access `http://localhost:5000` locally.

## 🌟 Recent Updates (May 2026)
*   **Vercel Cloud UI**: Split the frontend into a static shell (`vercel_app.py` / `main.js`) that uses CORS and ngrok bypass headers to communicate with your local machine.
*   **Live Cloud Progress**: Custom `fetch()`-based Server-Sent Events allow you to view live progress bars and translation output directly from the Vercel site.
*   **Smart Auto-Detection**: Integrated a new `fastapi` sidecar that handles multimodal language detection. If the source language is set to "Auto-Detect", the pipeline samples pages 4-5 and identifies the language before processing.
*   **Chinese Variant Disambiguation**: Enhanced detection logic that uses AI to distinguish between Simplified (`zho_Hans`) and Traditional (`zho_Hant`) Chinese text automatically.

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
