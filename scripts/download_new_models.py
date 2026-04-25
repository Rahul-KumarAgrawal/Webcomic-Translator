import os
from huggingface_hub import hf_hub_download
from pathlib import Path

_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
save_dir = _ROOT / "Pipeline Koharu" / "Detection and Layout" / "models"
save_dir.mkdir(parents=True, exist_ok=True)

models = [
    ("ogkalu/comic-text-and-bubble-detector", "detector.onnx"),
    ("kitsumed/yolov8m_seg-speech-bubble", "model.pt"),
    ("ogkalu/comic-text-segmenter-yolov8m", "comic-text-segmenter.pt"),
]

for repo, filename in models:
    print(f"Downloading {filename} from {repo}...")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(save_dir)
    )
    print(f"  Saved to {save_dir / filename}")

print("\nAll models downloaded!")
