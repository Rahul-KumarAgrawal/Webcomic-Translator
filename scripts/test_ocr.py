import sys, traceback
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='japan')
except Exception as e:
    with open('test_error.txt', 'w') as f:
        traceback.print_exc(file=f)
