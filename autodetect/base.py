from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self):
        self.name = ""
        self.requires_api_key = True
        self.is_free = False

    @abstractmethod
    def detect(self, image_bytes: bytes) -> dict:
        """
        Detect language from image bytes.
        Returns:
            dict: {
                "languages": [
                    { "name": "Japanese", "script": "Hiragana", "confidence": "high" }
                ],
                "hasText": bool,
                "summary": str,
                "engine": str
            }
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the engine is reachable and configured correctly."""
        pass
