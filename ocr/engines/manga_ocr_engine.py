from ..base import OCREngine
from typing import Set, Union
import numpy as np
from PIL import Image

class MangaOCREngine(OCREngine):
    def __init__(self):
        self.mocr = None
        
    def _get_model(self):
        from manga_ocr import MangaOcr
        if self.mocr is None:
            self.mocr = MangaOcr()
        return self.mocr

    def recognize(self, image: Union[np.ndarray, Image.Image], lang: str) -> str:
        if lang != 'ja': raise ValueError("MangaOCR only supports 'ja'")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._get_model()(image)
        
    def supported_languages(self) -> Set[str]:
        return {'ja'}
        
    def name(self) -> str: return 'mangaocr'
    
    def is_available(self) -> bool: return True
