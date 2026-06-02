import os
import glob

# The punctuation characters we want to ensure are not missed
PUNCTUATIONS = [
    '?', ',', '.', '!', '~', '…', '-', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '“', '”', '‘', '’',
    '......', '...', '!!', '?!', '!?', '⁉', '⁈', '⁇', '⁉️'
]

# Write a dedicated punctuation_dict.txt to all OCR directories
ocr_base_dir = r"d:\Webcomic #7\Pipeline Koharu\OCR"
cache_dirs = [
    r"d:\Webcomic #7\model\paddle_cache",
    r"d:\Webcomic #7\model\easyocr_cache"
]

if os.path.exists(ocr_base_dir):
    ocr_dirs = [os.path.join(ocr_base_dir, d) for d in os.listdir(ocr_base_dir) if os.path.isdir(os.path.join(ocr_base_dir, d))]
else:
    ocr_dirs = []

all_target_dirs = ocr_dirs + cache_dirs + [ocr_base_dir]

for d in all_target_dirs:
    if os.path.exists(d):
        dict_path = os.path.join(d, "punctuation_dict.txt")
        with open(dict_path, "w", encoding="utf-8") as f:
            for p in PUNCTUATIONS:
                f.write(p + "\n")
        print(f"Created {dict_path}")

# Additionally, for PPOCR-v5-onnx, let's make sure the single characters are in the main dictionaries
ppocr_dir = os.path.join(ocr_base_dir, "ppocr-v5-onnx")
if os.path.exists(ppocr_dir):
    dict_files = glob.glob(os.path.join(ppocr_dir, "*dict.txt"))
    for df in dict_files:
        if "punctuation_dict" in df:
            continue
        with open(df, "r", encoding="utf-8") as f:
            existing_chars = set(line.strip() for line in f.readlines())
        
        added = False
        with open(df, "a", encoding="utf-8") as f:
            for p in PUNCTUATIONS:
                # PaddleOCR dictionaries are character-by-character
                for char in p:
                    if char not in existing_chars:
                        f.write("\n" + char)
                        existing_chars.add(char)
                        added = True
        if added:
            print(f"Appended missing punctuation characters to {df}")
