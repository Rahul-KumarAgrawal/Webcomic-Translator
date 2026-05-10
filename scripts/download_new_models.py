import os
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
save_dir = _ROOT / "Pipeline Koharu" / "Detection and Layout" / "models"
ocr_dir = _ROOT / "Pipeline Koharu" / "OCR"
inpainting_dir = _ROOT / "models" / "inpainting"
detector_dir = _ROOT / "models" / "detector"

for d in [save_dir, ocr_dir, inpainting_dir, detector_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Base Detection Models
models = [
    ("ogkalu/comic-text-and-bubble-detector", "detector.onnx"),
    ("kitsumed/yolov8m_seg-speech-bubble", "model.pt"),
    ("ogkalu/comic-text-segmenter-yolov8m", "comic-text-segmenter.pt"),
]

# 2. Specialized Inpainters & Detectors (New)
print("--- [Elite Engines] Downloading specialized models ---")
# Mayo/Fashn-AI LaMa
try:
    print("Downloading LaMa (Fashn-AI)...")
    hf_hub_download(repo_id="fashn-ai/LaMa", filename="big-lama.pt", local_dir=str(inpainting_dir))
    src = inpainting_dir / "big-lama.pt"
    dst = inpainting_dir / "mayo_panel_cleaner.pt"
    if src.exists() and not dst.exists():
        os.rename(src, dst)
except Exception as e:
    print(f"  Warning: Failed to download LaMa: {e}")

# AOT-GAN
try:
    print("Downloading AOT-GAN...")
    snapshot_download(repo_id="mayocream/aot-inpainting", local_dir=str(inpainting_dir / "aot_gan"))
except Exception as e:
    print(f"  Warning: Failed to download AOT-Inpainting: {e}")

# Comic Text Detector (Precision Eyes)
try:
    print("Downloading Comic Text Detector (Precision Segmenter)...")
    hf_hub_download(repo_id="mayocream/comic-text-detector", filename="comictextdetector.pt.onnx", local_dir=str(detector_dir))
except Exception as e:
    print(f"  Warning: Failed to download comictextdetector: {e}")

# 3. OCR and Larger Repo Collections
repos = [
    ("ogkalu/pororo", "pororo"),
    ("ogkalu/ppocr-v5-torch", "ppocr-v5-torch"),
    ("ogkalu/yuzumarker-font-detection-onnx", "font-detection"),
    ("mayocream/speech-bubble-segmentation", "mayo-bubble-seg"),
    ("mayocream/comic-text-detector", "mayo-text-classic"),
    ("ogkalu/comic-text-segmenter-yolov8m", "ogkalu-text-stable"),
    ("ogkalu/comic-speech-bubble-detector-yolov8m", "ogkalu-bubble-stable"),
]

print("\n--- [Core Engines] Downloading base models ---")
for repo, filename in models:
    print(f"Downloading {filename} from {repo}...")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=str(save_dir))

for repo_id, folder_name in repos:
    if "segmentation" in repo_id or "yolo" in repo_id or "detector" in repo_id:
        target = save_dir / folder_name
    else:
        target = ocr_dir / folder_name
        
    print(f"Downloading full repo {repo_id} to {target}...")
    snapshot_download(repo_id=repo_id, local_dir=str(target))

print("\n[COMPLETE] All models and OCR engines are ready!")
