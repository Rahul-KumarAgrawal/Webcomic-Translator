import os
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
base_dir = _ROOT / "Pipeline Koharu"

models_to_download = [
    # (folder, type, repo_id, filename/None)
    ("Detection and Layout", "file", "ogkalu/comic-text-and-bubble-detector", "detector.onnx"),
    ("Detection and Layout", "file", "ogkalu/comic-text-segmenter-yolov8m", "comic-text-segmenter.pt"),
    ("Detection and Layout", "file", "ogkalu/comic-speech-bubble-detector-yolov8m", "comic-speech-bubble-detector.pt"),
    ("OCR", "repo", "PaddlePaddle/PaddleOCR-VL-1.5", None),
    ("OCR", "repo", "kha-white/manga-ocr-base", None),
    ("Inpainting", "repo", "ogkalu/aot-inpainting", None),
    ("Inpainting", "repo", "mayocream/lama-manga", None),
    ("Font Analysis", "repo", "fffonion/yuzumarker-font-detection", None),
]

print("Starting model downloads for Pipeline Koharu...\n")

for folder, dl_type, repo_id, filename in models_to_download:
    save_dir = base_dir / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if dl_type == "file":
            print(f"Downloading file {filename} from {repo_id}...")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(save_dir))
            print(f"SUCCESS: Downloaded to {save_dir}\n")
            
        elif dl_type == "repo":
            repo_name = repo_id.split('/')[-1]
            repo_save_dir = save_dir / repo_name
            repo_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading full repo {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=str(repo_save_dir))
            print(f"SUCCESS: Downloaded to {repo_save_dir}\n")
            
    except Exception as e:
        print(f"FAILURE: Failed to download {repo_id}. Error: {e}\n")

print("Finished downloading all models!")
