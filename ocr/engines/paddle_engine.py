from ..base import OCREngine
from typing import Set, Union
import numpy as np
from PIL import Image

class PaddleOCREngine(OCREngine):
    def __init__(self):
        self.models = {}
        self.lang_map = {'ja': 'japan', 'ko': 'korean', 'zh-CN': 'ch', 'zh-TW': 'chinese_cht'}
        
    def _get_model(self, lang: str):
        from paddleocr import PaddleOCR
        if lang not in self.models:
            self.models[lang] = PaddleOCR(use_angle_cls=True, lang=self.lang_map[lang])
        return self.models[lang]

    def recognize(self, image: Union[np.ndarray, Image.Image], lang: str) -> str:
        if lang not in self.lang_map: raise ValueError(f'Unsupported lang: {lang}')
        ocr = self._get_model(lang)
        result = ocr.ocr(np.array(image), cls=True)
        if not result or not result[0]: return ""
        return ' '.join([line[1][0] for line in result[0]])
        
    def supported_languages(self) -> Set[str]:
        return {'ja', 'ko', 'zh-CN', 'zh-TW'}
        
    def name(self) -> str: return 'paddleocr'
    
    def is_available(self) -> bool: return True
