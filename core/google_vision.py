"""
core/google_vision.py
Google Cloud Vision API wrapper for language detection and OCR.
"""

import base64
import json
import logging
import os
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class GoogleVisionDetector:
    """
    Uses Google Cloud Vision API to detect language and extract text from images.
    Requires a Service Account JSON key file.
    """
    def __init__(self, key_path: str):
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Google Vision key file not found at: {key_path}")
        
        self.key_path = key_path
        self._access_token = None
        self._token_expiry = 0

    def _get_access_token(self):
        """
        Retrieves an OAuth2 access token using the service account JSON.
        """
        import time
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        if self._access_token and time.time() < self._token_expiry:
            return self._access_token

        try:
            scopes = ['https://www.googleapis.com/auth/cloud-platform']
            creds = service_account.Credentials.from_service_account_file(self.key_path, scopes=scopes)
            creds.refresh(Request())
            self._access_token = creds.token
            self._token_expiry = creds.expiry.timestamp() if creds.expiry else time.time() + 3500
            return self._access_token
        except Exception as e:
            logger.error(f"Failed to get Google Vision access token: {e}")
            return None

    def detect_language(self, image_path: str) -> Optional[str]:
        """
        Identifies the dominant language in the image.
        Returns ISO-639-1 code (e.g., 'ja', 'zh', 'ko').
        """
        token = self._get_access_token()
        if not token:
            return None

        try:
            with open(image_path, "rb") as image_file:
                content = base64.b64encode(image_file.read()).decode('utf-8')

            url = "https://vision.googleapis.com/v1/images:annotate"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "requests": [
                    {
                        "image": {"content": content},
                        "features": [{"type": "TEXT_DETECTION"}]
                    }
                ]
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Extract detected languages
            responses = data.get("responses", [])
            if not responses or "textAnnotations" not in responses[0]:
                return None

            full_text_annotation = responses[0].get("fullTextAnnotation", {})
            pages = full_text_annotation.get("pages", [])
            if not pages:
                return None

            # Get the language with the highest confidence from the first page
            langs = pages[0].get("property", {}).get("detectedLanguages", [])
            if langs:
                # Return the most confident one
                best_lang = max(langs, key=lambda x: x.get("confidence", 0))
                return best_lang.get("languageCode")

            return None

        except Exception as e:
            logger.error(f"Google Vision API error: {e}")
            return None

# Mapping Google codes to project NLLB codes
GOOGLE_TO_NLLB = {
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "zh-CN": "zho_Hans",
    "zh-TW": "zho_Hant",
    "ko": "kor_Hang",
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
}
