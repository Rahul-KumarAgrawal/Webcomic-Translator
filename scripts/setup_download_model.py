"""
setup_download_model.py
Downloads NLLB-200 600M weights from HuggingFace to model/base/
Run by setup.bat — do not run this in parallel with the web app.
"""
import os
import sys

dest = os.path.join("model", "base")
print(f"  Downloading NLLB-200 600M to: {os.path.abspath(dest)}")
print("  This may take 5-15 minutes depending on your connection speed...")

try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="facebook/nllb-200-distilled-600M",
        local_dir=dest,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print("  Download complete!")
except ImportError:
    print("[ERROR] huggingface_hub not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="facebook/nllb-200-distilled-600M",
        local_dir=dest,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print("  Download complete!")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    sys.exit(1)
