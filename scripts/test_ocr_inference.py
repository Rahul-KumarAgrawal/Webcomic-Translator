import numpy as np
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='japan')
res = ocr.ocr(np.zeros((100, 100, 3), dtype=np.uint8))
print(f"RES: {res}")
