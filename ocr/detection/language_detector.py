import regex as re
from typing import Tuple
from .chinese_variant import ChineseVariantDetector

class LanguageDetector:
    def __init__(self):
        self.zh_detector = ChineseVariantDetector()
        
    def detect(self, text: str) -> Tuple[str, float]:
        if not text.strip(): return 'unknown', 0.0
        script = self.detect_script(text)
        if script == 'hangul': return 'ko', 0.9
        if script == 'kana': return 'ja', 0.9
        if script == 'han': return self.zh_detector.detect(text)
        return 'unknown', 0.0
        
    def detect_script(self, text: str) -> str:
        if re.search(r'\p{Hangul}', text): return 'hangul'
        if re.search(r'\p{Hiragana}|\p{Katakana}', text): return 'kana'
        if re.search(r'\p{Han}', text): return 'han'
        return 'unknown'
