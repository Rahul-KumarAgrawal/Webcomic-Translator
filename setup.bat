@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM ===========================================================================
REM  CBZ Translator — Setup Script
REM  Ensures CRLF line endings and UTF-8 encoding.
REM ===========================================================================

REM Suppress all "not on PATH" warnings globally
set PIP_NO_WARN_SCRIPT_LOCATION=1

cd /d "%~dp0"

echo.
echo  +--------------------------------------------------+
echo  ^|      CBZ Translator — Automated Setup           ^|
echo  ^|   (Waifu2x, EasyOCR ^& Batch Pipeline)        ^|
echo  +--------------------------------------------------+
echo.
echo [SETUP] Working directory: %CD%

REM -- Detect Python --
set "PYTHON="

if exist ".\python\python.exe" (
    set "PYTHON=.\python\python.exe"
    echo [SETUP] Using local Python: .\python\python.exe
    goto :python_found
)

if exist ".\venv\Scripts\python.exe" (
    set "PYTHON=.\venv\Scripts\python.exe"
    echo [SETUP] Using existing virtual environment: .\venv\
    goto :python_found
)

REM Try system Python
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%V in ('python --version 2^>^&1') do set "SYS_PY_VER=%%V"
    echo.
    echo  [SYSTEM PYTHON FOUND]
    echo  Version: !SYS_PY_VER!
    echo.
    echo  [1] Use system Python (creates ./venv/)
    echo  [2] Install standalone Python (downloads to ./python/)
    echo.
    set /p "PY_CHOICE=Enter choice (1 or 2): "
    if "!PY_CHOICE!"=="2" goto :install_standalone_python
    
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    set "PYTHON=.\venv\Scripts\python.exe"
    goto :python_found
)

REM Try python3
python3 --version >nul 2>&1
if not errorlevel 1 (
    echo [SETUP] Creating virtual environment with python3...
    python3 -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    set "PYTHON=.\venv\Scripts\python.exe"
    goto :python_found
)

echo [ERROR] Python not found! Download from https://www.python.org/
pause & exit /b 1

:install_standalone_python
echo [SETUP] Downloading standalone Python 3.10...
if not exist "python" mkdir python
set "PY_ZIP=python310.zip"
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile '%PY_ZIP%' -UseBasicParsing"
if errorlevel 1 (echo [ERROR] Download failed. & pause & exit /b 1)
powershell -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '.\python' -Force"
del /q "%PY_ZIP%"
if exist "python\python310._pth" (
    powershell -Command "(Get-Content 'python\python310._pth') -replace '#import site','import site' | Set-Content 'python\python310._pth'"
)
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing"
.\python\python.exe get-pip.py --quiet
del /q get-pip.py
set "PYTHON=.\python\python.exe"

:python_found
if not defined PYTHON (
    echo [ERROR] Python path resolution failed.
    pause & exit /b 1
)
for /f "tokens=*" %%V in ('"%PYTHON%" --version 2^>^&1') do echo [SETUP] Found: %%V

REM -- Create Folders --
echo [SETUP] Creating folder structure...
for %%D in (input output logs fonts backups core memory memory\global memory\series model model\base model\finetuned model\training_data model\cache web web\templates web\static config) do (
    if not exist "%%D" mkdir "%%D"
)
for %%P in (core memory model web) do (
    if not exist "%%P\__init__.py" type nul > "%%P\__init__.py"
)

REM -- Pip Setup --
echo [SETUP] Updating pip...
"%PYTHON%" -m pip install --upgrade pip --quiet

REM -- Check for Tesseract --
if not exist ".\Tesseract-OCR\tesseract.exe" (
    echo.
    echo  [!] WARNING: Tesseract-OCR not found in ./Tesseract-OCR/
    echo      The Symbol Fallback and Tesseract Engine will not work.
    echo      Download it here: https://github.com/UB-Mannheim/tesseract/wiki
    echo.
) else (
    echo [SETUP] Tesseract-OCR engine found.
)
"%PYTHON%" -m pip install --upgrade pip setuptools wheel

REM -- Check for Node.js (npx) for ngrok --
call npx --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [!] WARNING: Node.js (npx) not found!
    echo      The permanent ngrok tunnel in run_web.bat requires Node.js.
    echo      Download it here: https://nodejs.org/
    echo.
) else (
    echo [SETUP] Node.js (npx) found.
)

REM -- Phase 1: PyTorch --
echo [SETUP] Installing PyTorch (CUDA 12.6)...
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo [WARNING] CUDA install failed, trying default...
    "%PYTHON%" -m pip install torch torchvision
)

REM -- Phase 2: Requirements --
echo [SETUP] Installing dependencies from requirements.txt...
"%PYTHON%" -m pip install -r requirements.txt

REM -- Phase 3: NVIDIA Binaries & ML Components --
echo [SETUP] Installing NVIDIA runtime binaries...
"%PYTHON%" -m pip install nvidia-cudnn-cu11==8.9.4.19 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cufft-cu11 nvidia-cuda-nvrtc-cu11 2>nul

echo [SETUP] Installing MIT/OCR components...
"%PYTHON%" -m pip install torch-summary einops kornia timm open_clip_torch safetensors ctranslate2 onnxruntime-gpu
"%PYTHON%" -m pip install manga-ocr
"%PYTHON%" -m pip install paddlepaddle-gpu==2.6.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ paddleocr==2.8.1

REM -- Phase 4: Utilities & Elite Engines --
echo [SETUP] Installing image processing and Elite Engines...
"%PYTHON%" -m pip install pororo-ocr wget pytesseract
"%PYTHON%" -m pip install scikit-image opencv-python Pillow pillow-avif-plugin ImageHash pyclipper shapely freetype-py git+https://github.com/facebookresearch/segment-anything.git
"%PYTHON%" -m pip install transformers sentencepiece accelerate openai==1.63.0 tiktoken deepl groq google-genai py3langid==0.2.2 langdetect langcodes easyocr editdistance
"%PYTHON%" -m pip install segmentation-models-pytorch safetensors
"%PYTHON%" -m pip install google-genai pydantic>=2.9.0 httpx>=0.28.1
"%PYTHON%" -m pip install omegaconf pydantic>=2.9.0 marshmallow "protobuf<6.0.0" aiohttp aiofiles aioshutil httpx>=0.28.1 requests websockets nest-asyncio uvicorn fastapi python-multipart
"%PYTHON%" -m pip install arabic-reshaper python-bidi pyhyphen tqdm colorama rich regex networkx pandas tensorboardX cryptography python-dotenv backports.cached-property cython bitsandbytes huggingface_hub datasets peft>=0.9.0

REM -- Phase 5: Rust Components --
echo [SETUP] Installing MIT Rust components...
"%PYTHON%" -m pip install --extra-index-url https://frederik-uni.github.io/manga-image-translator-rust/python/wheels/simple/ rusty-manga-image-translator 2>nul

REM -- Phase 6: Database & Training Files --
echo [SETUP] Initializing database...
"%PYTHON%" scripts\setup_init_db.py
if not exist "model\training_data\approved_pairs.jsonl" type nul > "model\training_data\approved_pairs.jsonl"
if not exist "model\training_data\rejected_pairs.jsonl" type nul > "model\training_data\rejected_pairs.jsonl"

REM -- Phase 7: Model Downloads --
echo [SETUP] Checking models...
if not exist "models\inpainting\mayo_panel_cleaner.pt" (
    echo [SETUP] Downloading specialized inpainting models...
    "%PYTHON%" scripts\download_new_models.py
) else if not exist "Pipeline Koharu\Detection and Layout\models\ogkalu-text-stable" (
    echo [SETUP] Downloading detection models...
    "%PYTHON%" scripts\download_new_models.py
) else if not exist "Pipeline Koharu\OCR\font-detection\font-detector_int8.onnx" (
    echo [SETUP] Downloading new font-detection models...
    "%PYTHON%" scripts\download_new_models.py
) else if not exist "Pipeline Koharu\Inpainting\aot-inpainting\aot.onnx" (
    echo [SETUP] Downloading AOT-Inpainting ONNX...
    "%PYTHON%" scripts\download_new_models.py
) else if not exist "Pipeline Koharu\Detection and Layout\models\detector_int8.onnx" (
    echo [SETUP] Downloading combined models...
    "%PYTHON%" scripts\download_new_models.py
)

REM -- Final Verification --
echo [SETUP] Verifying installation...
"%PYTHON%" -c "import torch; print('PyTorch OK'); import flask; print('Flask OK'); import cv2; print('OpenCV OK')"
if errorlevel 1 (
    echo [ERROR] Verification failed. Some packages are missing.
) else (
    echo [SETUP] All core packages verified.
)

echo %PYTHON%> .python_path.txt
echo [SETUP] Setup Complete!
pause
