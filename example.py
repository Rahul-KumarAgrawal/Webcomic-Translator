from ocr.pipeline import TranslationPipeline
import numpy as np

if __name__ == '__main__':
    pipeline = TranslationPipeline()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    print("Test 1: Auto detect with easyocr")
    try:
        res = pipeline.run(img, 'easyocr', 'auto')
        print(res)
    except Exception as e:
        print("Error:", e)
