import os
import urllib.request
import tarfile

_ROOT = os.path.dirname(os.path.abspath(__file__))
PADDLE_CACHE = os.path.join(_ROOT, "model", "paddle_cache", "whl")

MODELS = [
    {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar",
        "dest": os.path.join(PADDLE_CACHE, "det", "ml"),
        "filename": "Multilingual_PP-OCRv3_det_infer.tar",
        "extracted_folder": "Multilingual_PP-OCRv3_det_infer"
    },
    {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar",
        "dest": os.path.join(PADDLE_CACHE, "rec", "japan"),
        "filename": "japan_PP-OCRv4_rec_infer.tar",
        "extracted_folder": "japan_PP-OCRv4_rec_infer"
    },
    {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar",
        "dest": os.path.join(PADDLE_CACHE, "rec", "korean"),
        "filename": "korean_PP-OCRv4_rec_infer.tar",
        "extracted_folder": "korean_PP-OCRv4_rec_infer"
    },
    {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "dest": os.path.join(PADDLE_CACHE, "det", "en"),
        "filename": "en_PP-OCRv3_det_infer.tar",
        "extracted_folder": "en_PP-OCRv3_det_infer"
    },
    {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "dest": os.path.join(PADDLE_CACHE, "rec", "en"),
        "filename": "en_PP-OCRv4_rec_infer.tar",
        "extracted_folder": "en_PP-OCRv4_rec_infer"
    }
]

def download_and_extract():
    print("Downloading PaddleOCR models...")
    for model in MODELS:
        os.makedirs(model["dest"], exist_ok=True)
        tar_path = os.path.join(model["dest"], model["filename"])
        
        extracted_folder = os.path.join(model["dest"], model["extracted_folder"])
        if os.path.exists(extracted_folder):
            print(f"Skipping {model['filename']} - already extracted.")
            continue
            
        print(f"Downloading {model['filename']}...")
        try:
            urllib.request.urlretrieve(model["url"], tar_path)
            print(f"Extracting {model['filename']}...")
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=model["dest"])
            os.remove(tar_path)
        except Exception as e:
            print(f"Failed to download/extract {model['filename']}: {e}")

if __name__ == "__main__":
    download_and_extract()
