from ..base import OCREngine
from typing import Set, Union
import numpy as np
from PIL import Image

class EasyOCREngine(OCREngine):
    def __init__(self):
        self.readers = {}
        self.lang_map = {'ja': 'ja', 'ko': 'ko', 'zh-CN': 'ch_sim', 'zh-TW': 'ch_tra'}
        
    def _get_reader(self, lang: str):
        import easyocr
        if lang not in self.readers:
            self.readers[lang] = easyocr.Reader([self.lang_map[lang]], gpu=True)
        return self.readers[lang]

    def recognize(self, image: Union[np.ndarray, Image.Image], lang: str) -> str:
        if lang not in self.lang_map: raise ValueError(f'Unsupported lang: {lang}')
        reader = self._get_reader(lang)
        result = reader.readtext(np.array(image), detail=0)
        return ' '.join(result)
        
    def supported_languages(self) -> Set[str]:
        return {'ja', 'ko', 'zh-CN', 'zh-TW'}
        
    def name(self) -> str: return 'easyocr'
    
    def is_available(self) -> bool: return True
