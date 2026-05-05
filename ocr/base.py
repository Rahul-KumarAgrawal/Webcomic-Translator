from abc import ABC, abstractmethod
from typing import Set, Union
import numpy as np
from PIL import Image

class OCREngine(ABC):
    @abstractmethod
    def recognize(self, image: Union[np.ndarray, Image.Image], lang: str) -> str: ...
    
    @abstractmethod
    def supported_languages(self) -> Set[str]: ...
    
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def is_available(self) -> bool: ...
