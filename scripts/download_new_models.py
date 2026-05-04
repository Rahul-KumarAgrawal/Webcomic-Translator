import os
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
save_dir = _ROOT / "Pipeline Koharu" / "Detection and Layout" / "models"
ocr_dir = _ROOT / "Pipeline Koharu" / "OCR"
save_dir.mkdir(parents=True, exist_ok=True)
ocr_dir.mkdir(parents=True, exist_ok=True)

models = [
    ("ogkalu/comic-text-and-bubble-detector", "detector.onnx"),
    ("kitsumed/yolov8m_seg-speech-bubble", "model.pt"),
    ("ogkalu/comic-text-segmenter-yolov8m", "comic-text-segmenter.pt"),
]

repos = [
    ("ogkalu/pororo", "pororo"),
    ("ogkalu/ppocr-v5-torch", "ppocr-v5-torch"),
    ("ogkalu/yuzumarker-font-detection-onnx", "font-detection"),
    ("mayocream/speech-bubble-segmentation", "mayo-bubble-seg"),
    ("mayocream/comic-text-detector", "mayo-text-classic"),
    ("ogkalu/comic-text-segmenter-yolov8m", "ogkalu-text-stable"),
    ("ogkalu/comic-speech-bubble-detector-yolov8m", "ogkalu-bubble-stable"),
]

# Specifically ensure ONNX versions are pulled for Mayo
def download_mayo_onnx():
    print("Ensuring high-speed ONNX versions for Mayo...")
    # mayo-text-classic -> dbnet.onnx
    try:
        hf_hub_download(repo_id="mayocream/comic-text-detector", filename="dbnet.onnx", local_dir=str(save_dir / "mayo-text-classic"))
    except:
        print("  Failed to get dbnet.onnx specifically, will try full snapshot.")
        
    # mayo-bubble-seg -> model.onnx
    try:
        hf_hub_download(repo_id="mayocream/speech-bubble-segmentation", filename="model.onnx", local_dir=str(save_dir / "mayo-bubble-seg"))
    except:
        print("  Failed to get model.onnx specifically, will try full snapshot.")

for repo, filename in models:
    print(f"Downloading {filename} from {repo}...")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=str(save_dir))
    print(f"  Saved to {save_dir / filename}")

for repo_id, folder_name in repos:
    # Route detection models to the detection folder, OCR to OCR folder
    if "segmentation" in repo_id or "yolo" in repo_id or "detector" in repo_id:
        target = save_dir / folder_name
    else:
        target = ocr_dir / folder_name
        
    print(f"Downloading full repo {repo_id} to {target}...")
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"  Repo {folder_name} ready.")

download_mayo_onnx()

print("\nAll models and OCR engines downloaded!")
