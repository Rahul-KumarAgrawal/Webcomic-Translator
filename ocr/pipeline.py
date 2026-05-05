import logging
from typing import Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
from .config import Config
from .detection.language_detector import LanguageDetector
from .engines.easyocr_engine import EasyOCREngine
from .engines.tesseract_engine import TesseractOCREngine
from .engines.paddle_engine import PaddleOCREngine
from .engines.manga_ocr_engine import MangaOCREngine
from .utils import link_dlls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ocr_pipeline')

@dataclass
class OCRResult:
    text: str
    detected_lang: str
    engine_used: str
    confidence: float
    fallback_occurred: bool
    logs: list

class TranslationPipeline:
    def __init__(self, config=None):
        link_dlls()
        self.config = config or Config()
        self.detector = LanguageDetector()
        self.engines = {
            'easyocr': EasyOCREngine(),
            'tesseract': TesseractOCREngine(),
            'paddleocr': PaddleOCREngine(),
            'mangaocr': MangaOCREngine()
        }
        
    def run(self, image: Union[np.ndarray, Image.Image], user_engine: str, user_lang: str) -> OCRResult:
        logs = []
        fallback_occurred = False
        confidence = 1.0
        
        if user_lang == 'auto':
            logger.info("Running detection pass with PaddleOCR")
            rough_text = self.engines['paddleocr'].recognize(image, 'zh-CN')  # Using ch model for rough detection
            user_lang, confidence = self.detector.detect(rough_text)
            logger.info(f"Detected language: {user_lang} (confidence={confidence})")
            logs.append(f"Detected language: {user_lang} (confidence={confidence})")
        
        engine_name = user_engine
        
        # Override rules
        if user_lang == 'ja' and user_engine == 'mangaocr':
            engine_name = 'mangaocr'
        elif user_lang == 'ja' and user_engine != 'mangaocr':
            logger.info("Suggestion: MangaOCR is recommended for Japanese.")
            logs.append("Suggestion: MangaOCR is recommended for Japanese.")
            
        engine = self.engines.get(engine_name)
        if engine and user_lang not in engine.supported_languages():
            logger.warning(f"Engine {engine_name} doesn't support {user_lang}, falling back.")
            logs.append(f"Engine {engine_name} doesn't support {user_lang}, falling back.")
            fallback_occurred = True
            for alt_name in self.config.engine_priority:
                alt_engine = self.engines[alt_name]
                if user_lang in alt_engine.supported_languages():
                    engine = alt_engine
                    engine_name = alt_name
                    break
                    
        text = engine.recognize(image, user_lang)
        logger.info(f"Final OCR completed with {engine_name}")
        logs.append(f"Final OCR completed with {engine_name}")
        
        return OCRResult(text, user_lang, engine_name, confidence, fallback_occurred, logs)
