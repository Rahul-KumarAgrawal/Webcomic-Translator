@echo off
chcp 65001 >nul
REM ═══════════════════════════════════════════════════════════════════════════
REM  CBZ Translator — Setup Script (Comprehensive)
REM  Run once to install ALL dependencies and initialise the project.
REM  Safe to re-run: skips anything already installed / downloaded.
REM  Supports: Local ./python/, system Python, or auto-created ./venv/
REM ═══════════════════════════════════════════════════════════════════════════
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║      CBZ Translator — Automated Setup           ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo [SETUP] Working directory: %CD%
echo.

REM ── Detect Python ─────────────────────────────────────────────────────────
REM Priority 1: Local ./python/ folder (portable install)
REM Priority 2: Active ./venv/ virtual environment
REM Priority 3: System python — if found, create ./venv/ for isolation

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
    echo  ╔══════════════════════════════════════════════════╗
    echo  ║           Python Installation Choice            ║
    echo  ╠══════════════════════════════════════════════════╣
    echo  ║  A Python installation was found on your PC:    ║
    echo  ║  !SYS_PY_VER!
    echo  ║                                                  ║
    echo  ║  [1] Use your current system Python             ║
    echo  ║      (creates an isolated ./venv/ folder)       ║
    echo  ║                                                  ║
    echo  ║  [2] Install standalone Python for this project ║
    echo  ║      (downloads into ./python/ — self-contained)║
    echo  ╚══════════════════════════════════════════════════╝
    echo.
    set /p "PY_CHOICE=Enter choice (1 or 2): "
    if "!PY_CHOICE!"=="2" goto :install_standalone_python
    echo [SETUP] Using system Python. Creating isolated virtual environment at .\venv\ ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    set "PYTHON=.\venv\Scripts\python.exe"
    echo [SETUP] Virtual environment created. Using: .\venv\Scripts\python.exe
    goto :python_found
)

REM Try python3 as well (some installations)
python3 --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%V in ('python3 --version 2^>^&1') do set "SYS_PY_VER=%%V"
    echo.
    echo  ╔══════════════════════════════════════════════════╗
    echo  ║           Python Installation Choice            ║
    echo  ╠══════════════════════════════════════════════════╣
    echo  ║  A Python installation was found on your PC:    ║
    echo  ║  !SYS_PY_VER!
    echo  ║                                                  ║
    echo  ║  [1] Use your current system Python             ║
    echo  ║      (creates an isolated ./venv/ folder)       ║
    echo  ║                                                  ║
    echo  ║  [2] Install standalone Python for this project ║
    echo  ║      (downloads into ./python/ — self-contained)║
    echo  ╚══════════════════════════════════════════════════╝
    echo.
    set /p "PY_CHOICE=Enter choice (1 or 2): "
    if "!PY_CHOICE!"=="2" goto :install_standalone_python
    echo [SETUP] Using system Python3. Creating isolated virtual environment at .\venv\ ...
    python3 -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    set "PYTHON=.\venv\Scripts\python.exe"
    echo [SETUP] Virtual environment created. Using: .\venv\Scripts\python.exe
    goto :python_found
)

echo [ERROR] Python not found!
echo         - If you have Python installed, make sure it is added to PATH.
echo         - Download Python 3.10+: https://www.python.org/downloads/
echo         - During install, tick "Add Python to PATH"
pause & exit /b 1

:install_standalone_python
REM ── Download portable Python 3.10 into ./python/ ──────────────────────────
echo.
echo [SETUP] Downloading standalone Python 3.10 into .\python\ ...
echo [SETUP] This will take a few minutes depending on your internet speed.
echo.

if not exist "python" mkdir python

REM Download Python 3.10 embeddable zip
set "PY_ZIP=python310.zip"
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile '%PY_ZIP%' -UseBasicParsing"
if errorlevel 1 (
    echo [ERROR] Failed to download Python. Check your internet connection.
    pause & exit /b 1
)

REM Extract
powershell -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '.\python' -Force"
del /q "%PY_ZIP%"

REM Enable site-packages in the embedded Python
set "PTH_FILE=python\python310._pth"
if exist "%PTH_FILE%" (
    powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site','import site' | Set-Content '%PTH_FILE%'"
)

REM Download & install pip
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing"
.\python\python.exe get-pip.py --quiet
del /q get-pip.py

set "PYTHON=.\python\python.exe"
echo [SETUP] Standalone Python installed. Using: .\python\python.exe

:python_found
for /f "tokens=*" %%V in ('"%PYTHON%" --version 2^>^&1') do echo [SETUP] Found: %%V
echo.

REM ── Create required folders ────────────────────────────────────────────────
echo [SETUP] Creating folder structure...
for %%D in (input output logs fonts backups ^
            core memory memory\global memory\series ^
            model model\base model\finetuned model\training_data model\cache ^
            web web\templates web\static config) do (
    if not exist "%%D" mkdir "%%D"
)

REM ── Create __init__.py files ───────────────────────────────────────────────
for %%P in (core memory model web) do (
    if not exist "%%P\__init__.py" type nul > "%%P\__init__.py"
)

REM ── Ensure pip is available ────────────────────────────────────────────────
echo [SETUP] Checking pip...
"%PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [SETUP] pip not found — bootstrapping with ensurepip...
    "%PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] Could not bootstrap pip. Please reinstall Python with pip enabled.
        pause & exit /b 1
    )
    echo [SETUP] pip bootstrapped successfully.
) else (
    echo [SETUP] pip is available.
)

REM ── Upgrade pip + setuptools + wheel ──────────────────────────────────────
echo [SETUP] Upgrading pip, setuptools, wheel...
"%PYTHON%" -m pip install --upgrade pip setuptools wheel --quiet 2>nul

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 1: PyTorch with CUDA 12.6
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 1: PyTorch ──────────────────────────────────────────
"%PYTHON%" -c "import torch; print(f'[SETUP] PyTorch {torch.__version__} already installed (CUDA: {torch.cuda.is_available()})')" 2>nul
if errorlevel 1 (
    echo [SETUP] Installing PyTorch with CUDA 12.6 — this may take several minutes...
    "%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 (
        echo [WARNING] PyTorch CUDA 12.6 installation failed. Trying default...
        "%PYTHON%" -m pip install torch torchvision
    )
)

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 2: Core project dependencies (requirements.txt)
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 2: Core project dependencies ───────────────────────
echo [SETUP] Installing from requirements.txt...
"%PYTHON%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Some core packages may have failed. Re-run setup.bat to retry.
)

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 3: MIT dependencies (all-in-one)
REM  Installs everything MIT needs: OCR, detection, inpainting, translation
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 3: MIT dependencies (comprehensive) ────────────────

REM -- ML / Compute --
"%PYTHON%" -m pip install torch-summary einops kornia timm open_clip_torch safetensors --quiet 2>nul
"%PYTHON%" -m pip install ctranslate2 --quiet 2>nul
"%PYTHON%" -m pip install onnxruntime --quiet 2>nul

REM -- OCR --
"%PYTHON%" -m pip install manga-ocr --quiet 2>nul
"%PYTHON%" -m pip install paddlepaddle-gpu==3.0.0b2 paddleocr==3.5.0 --quiet 2>nul

REM -- NVIDIA GPU Binaries (cuDNN 8, CUDA 11) --
echo [SETUP] Installing NVIDIA runtime binaries for GPU acceleration...
"%PYTHON%" -m pip install nvidia-cudnn-cu11==8.9.4.19 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cufft-cu11 nvidia-cuda-nvrtc-cu11 --quiet 2>nul

REM -- Image processing --
"%PYTHON%" -m pip install scikit-image opencv-python Pillow ImageHash --quiet 2>nul
"%PYTHON%" -m pip install pyclipper shapely --quiet 2>nul
"%PYTHON%" -m pip install freetype-py --quiet 2>nul
"%PYTHON%" -m pip install git+https://github.com/facebookresearch/segment-anything.git --quiet 2>nul

REM -- NLP / Translation --
"%PYTHON%" -m pip install transformers sentencepiece accelerate --quiet 2>nul
"%PYTHON%" -m pip install openai==1.63.0 tiktoken --quiet 2>nul
"%PYTHON%" -m pip install deepl groq google-genai --quiet 2>nul
"%PYTHON%" -m pip install py3langid==0.2.2 langdetect langcodes easyocr --quiet 2>nul
"%PYTHON%" -m pip install editdistance --quiet 2>nul

REM -- Config / serialization --
"%PYTHON%" -m pip install omegaconf pydantic==2.5.0 marshmallow --quiet 2>nul
"%PYTHON%" -m pip install protobuf<6.0.0 --quiet 2>nul

REM -- Async / networking --
"%PYTHON%" -m pip install aiohttp aiofiles aioshutil --quiet 2>nul
"%PYTHON%" -m pip install httpx==0.27.2 requests websockets --quiet 2>nul
"%PYTHON%" -m pip install nest-asyncio --quiet 2>nul

REM -- Web / API --
"%PYTHON%" -m pip install uvicorn fastapi python-multipart --quiet 2>nul

REM -- Text rendering / language --
"%PYTHON%" -m pip install arabic-reshaper python-bidi pyhyphen --quiet 2>nul

REM -- Utilities --
"%PYTHON%" -m pip install tqdm colorama rich regex --quiet 2>nul
"%PYTHON%" -m pip install networkx pandas tensorboardX --quiet 2>nul
"%PYTHON%" -m pip install cryptography python-dotenv --quiet 2>nul
"%PYTHON%" -m pip install backports.cached-property --quiet 2>nul
"%PYTHON%" -m pip install cython --quiet 2>nul

REM -- Quantization --
"%PYTHON%" -m pip install bitsandbytes --quiet 2>nul

REM -- HuggingFace --
"%PYTHON%" -m pip install huggingface_hub datasets --quiet 2>nul

REM -- LoRA fine-tuning --
"%PYTHON%" -m pip install peft>=0.9.0 --quiet 2>nul

REM -- Rust components for MIT --
echo [SETUP] Installing MIT Rust components...
"%PYTHON%" -m pip install --extra-index-url https://frederik-uni.github.io/manga-image-translator-rust/python/wheels/simple/ rusty-manga-image-translator --quiet 2>nul
if errorlevel 1 (
    echo [WARNING] rusty-manga-image-translator failed to install — MIT may still work in Python mode.
)

REM -- pydensecrf --
echo [SETUP] Installing pydensecrf (mask refinement)...
"%PYTHON%" -c "import pydensecrf" 2>nul
if errorlevel 1 (
    "%PYTHON%" -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git --quiet 2>nul
    if errorlevel 1 (
        echo [WARNING] pydensecrf failed — mask refinement will be limited.
        echo [WARNING] You may need Visual C++ Build Tools first.
    )
)

REM -- numpy pin (easyocr may upgrade numpy to 2.x — force pin back to 1.26.4) --
echo [SETUP] Pinning numpy to 1.26.4 for PaddleOCR/MIT compatibility...
"%PYTHON%" -m pip install "numpy==1.26.4" --force-reinstall --quiet 2>nul

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 4: Project-specific dependencies
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 4: Project-specific dependencies ───────────────────

"%PYTHON%" -m pip install Flask>=3.0.0 Flask-CORS>=4.0.0 --quiet 2>nul
"%PYTHON%" -m pip install PyYAML>=6.0.1 --quiet 2>nul
"%PYTHON%" -m pip install win10toast plyer --quiet 2>nul
"%PYTHON%" -m pip install deepl>=1.16.0 --quiet 2>nul
"%PYTHON%" -m pip install langdetect>=1.0.9 --quiet 2>nul

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 5: Database init
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 5: Database init ───────────────────────────────────
"%PYTHON%" scripts\setup_init_db.py
if errorlevel 1 (
    echo [WARNING] Database init had issues — may already exist.
)

REM ── Create empty training data files if missing ────────────────────────────
if not exist "model\training_data\approved_pairs.jsonl" (
    type nul > "model\training_data\approved_pairs.jsonl"
    echo [SETUP] Created approved_pairs.jsonl
)
if not exist "model\training_data\rejected_pairs.jsonl" (
    type nul > "model\training_data\rejected_pairs.jsonl"
    echo [SETUP] Created rejected_pairs.jsonl
)

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 6: Download Pipeline Koharu Models
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 6: Pipeline Koharu Models ──────────────────────────
if exist "Pipeline Koharu\Detection and Layout\detector.onnx" (
    echo [SETUP] Koharu models appear to be downloaded — skipping.
) else (
    echo [SETUP] Downloading Pipeline Koharu Models...
    "%PYTHON%" scripts\download_koharu_models.py
    if errorlevel 1 (
        echo [WARNING] Koharu models download failed. Re-run setup.bat later to retry.
    )
)

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 7: Additional Detection Models
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 7: Additional Detection Models ─────────────────────
if exist "Pipeline Koharu\Detection and Layout\models\comic-text-segmenter.pt" (
    echo [SETUP] Additional detection models appear to be downloaded — skipping.
) else (
    echo [SETUP] Downloading additional detection models...
    "%PYTHON%" scripts\download_new_models.py
    if errorlevel 1 (
        echo [WARNING] Additional detection models download failed.
    )
)

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 8: PaddleOCR Multilingual Models
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 8: PaddleOCR Multilingual Models ───────────────────
if exist "model\paddle_cache\whl\det\ml\Multilingual_PP-OCRv3_det_infer" (
    echo [SETUP] PaddleOCR models appear to be downloaded — skipping.
) else (
    echo [SETUP] Downloading PaddleOCR Multilingual Models...
    "%PYTHON%" scripts\setup_download_paddle.py
    if errorlevel 1 (
        echo [WARNING] PaddleOCR models download failed.
    )
)

REM ══════════════════════════════════════════════════════════════════════════
REM  PHASE 9: Final verification
REM ══════════════════════════════════════════════════════════════════════════
echo.
echo [SETUP] ── Phase 9: Verifying all dependencies ──────────────────────
set "FAIL=0"
set "WARN=0"

echo [SETUP] Checking critical dependencies...

"%PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>nul
if errorlevel 1 (
    echo   [MISSING] PyTorch with CUDA
    set "FAIL=1"
) else (
    for /f "tokens=*" %%V in ('"%PYTHON%" -c "import torch; print(f'torch {torch.__version__} CUDA {torch.version.cuda}')" 2^>^&1') do echo   [OK] %%V
)

for %%M in (yaml flask flask_cors PIL numpy sentencepiece transformers peft deepl langdetect) do (
    "%PYTHON%" -c "import %%M" 2>nul
    if errorlevel 1 (
        echo   [MISSING] %%M
        set "FAIL=1"
    ) else (
        echo   [OK] %%M
    )
)

echo.
echo [SETUP] Checking MIT pipeline dependencies...
for %%M in (cv2 skimage einops kornia timm ctranslate2 manga_ocr omegaconf pydantic aiohttp openai shapely pyclipper) do (
    "%PYTHON%" -c "import %%M" 2>nul
    if errorlevel 1 (
        echo   [MISSING] %%M
        set "WARN=1"
    ) else (
        echo   [OK] %%M
    )
)

"%PYTHON%" -c "import pydensecrf" 2>nul
if errorlevel 1 (
    echo   [MISSING] pydensecrf (mask refinement)
    set "WARN=1"
) else (
    echo   [OK] pydensecrf
)

echo.
if "!FAIL!"=="1" (
    echo [ERROR] Critical dependencies are missing! The project will NOT work.
    echo [ERROR] Try running setup.bat again or install the missing packages manually.
)
if "!WARN!"=="1" (
    if "!FAIL!"=="0" (
        echo [WARNING] Some optional/MIT dependencies are missing.
        echo [WARNING] Core translation will work, but MIT pipeline may have issues.
    )
)
if "!FAIL!"=="0" if "!WARN!"=="0" (
    echo   All dependencies verified successfully!
)

REM ── Save which Python was resolved ────────────────────────────────────────
echo %PYTHON%> .python_path.txt

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║              Setup Complete!                     ║
echo  ╠══════════════════════════════════════════════════╣
echo  ║  Run run_web.bat   to launch the web UI         ║
echo  ║  Run run_batch.bat for batch processing         ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause
