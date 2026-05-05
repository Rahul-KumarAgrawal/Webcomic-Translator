from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    engine_priority: List[str] = None
    default_unknown_lang: str = 'ja'
    
    def __post_init__(self):
        if self.engine_priority is None:
            self.engine_priority = ['paddleocr', 'easyocr', 'tesseract', 'mangaocr']
