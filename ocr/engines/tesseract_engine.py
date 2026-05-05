from ..base import OCREngine
from typing import Set, Union
import numpy as np
from PIL import Image

class TesseractOCREngine(OCREngine):
    def __init__(self):
        self.lang_map = {'ja': 'jpn', 'ko': 'kor', 'zh-CN': 'chi_sim', 'zh-TW': 'chi_tra'}
        
    def recognize(self, image: Union[np.ndarray, Image.Image], lang: str) -> str:
        import pytesseract
        if lang not in self.lang_map: raise ValueError(f'Unsupported lang: {lang}')
        return pytesseract.image_to_string(image, lang=self.lang_map[lang]).strip()
        
    def supported_languages(self) -> Set[str]:
        return {'ja', 'ko', 'zh-CN', 'zh-TW'}
        
    def name(self) -> str: return 'tesseract'
    
    def is_available(self) -> bool: return True
