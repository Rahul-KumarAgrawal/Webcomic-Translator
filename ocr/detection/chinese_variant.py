from typing import Tuple
from .char_sets import SIMPLIFIED_ONLY_CHARS, TRADITIONAL_ONLY_CHARS
import opencc

class ChineseVariantDetector:
    def __init__(self):
        self.s2t = opencc.OpenCC('s2t')
        self.t2s = opencc.OpenCC('t2s')
        
    def detect(self, text: str) -> Tuple[str, float]:
        s_count = sum(1 for c in text if c in SIMPLIFIED_ONLY_CHARS)
        t_count = sum(1 for c in text if c in TRADITIONAL_ONLY_CHARS)
        if s_count > t_count: return 'zh-CN', 0.8
        if t_count > s_count: return 'zh-TW', 0.8
        
        # opencc roundtrip
        if self.t2s.convert(text) == text: return 'zh-CN', 0.6
        if self.s2t.convert(text) == text: return 'zh-TW', 0.6
        
        return 'zh-CN', 0.5
