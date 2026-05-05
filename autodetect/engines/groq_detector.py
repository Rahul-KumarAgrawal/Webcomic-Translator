import os
import base64
import json
import requests
from autodetect.base import BaseDetector

class GroqDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.name = "groq"
        self.requires_api_key = True
        self.is_free = False
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _get_api_key(self):
        return os.environ.get("GROQ_API_KEY", "")

    def detect(self, image_bytes: bytes) -> dict:
        api_key = self._get_api_key()
        if not api_key:
            return {
                "languages": [],
                "hasText": False,
                "summary": "Groq API key not configured",
                "engine": self.name,
                "error": "Missing API Key"
            }

        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        prompt = (
            "Analyze the following image and detect the primary language(s) of the text. "
            "Return ONLY a JSON object with the following structure: "
            "{\n"
            "  \"languages\": [\n"
            "    { \"name\": \"Language Name\", \"script\": \"Script Name\", \"confidence\": \"high/medium/low\" }\n"
            "  ],\n"
            "  \"hasText\": true,\n"
            "  \"summary\": \"Brief description of what you see\",\n"
            "  \"engine\": \"groq\"\n"
            "}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            return {
                "languages": [],
                "hasText": False,
                "summary": f"Error: {str(e)}",
                "engine": self.name,
                "error": str(e)
            }

    def test_connection(self) -> bool:
        api_key = self._get_api_key()
        if not api_key:
            return False
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Minimal test call (list models or a small completion)
        try:
            # We can try to list models or just a tiny chat completion without image
            test_payload = {
                "model": "llama3-8b-8192", # Use a cheaper model for connection test if possible, or same
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            resp = requests.post(self.url, headers=headers, json=test_payload, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False
