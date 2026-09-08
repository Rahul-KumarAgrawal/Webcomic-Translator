import os
import importlib
import inspect
import json
from pathlib import Path
from autodetect.base import BaseDetector

class DetectorManager:
    def __init__(self, config_path=None):
        self.engines = {}
        self.active_engine_name = None
        self._root = Path(__file__).parent
        self.config_path = config_path or self._root / "config.json"
        self.discover_engines()
        self.load_config()

    def discover_engines(self):
        engines_dir = self._root / "engines"
        if not engines_dir.exists():
            return

        for file in engines_dir.glob("*.py"):
            if file.name == "__init__.py":
                continue
            
            module_name = f"autodetect.engines.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                # Reload to ensure we pick up changes if any
                importlib.reload(module)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseDetector) and obj is not BaseDetector:
                        engine_instance = obj()
                        self.engines[engine_instance.name] = engine_instance
            except Exception as e:
                print(f"Failed to load engine from {file}: {e}")

    def list_engines(self):
        result = []
        for name, engine in self.engines.items():
            # Check if API key is present in env if required
            env_key = f"{name.upper()}_API_KEY"
            is_configured = True
            fallback_key = "GROQ_API_KEY" if name.startswith("groq") else None
            if engine.requires_api_key and not (os.environ.get(env_key) or (fallback_key and os.environ.get(fallback_key))):
                is_configured = False
            
            result.append({
                "name": name,
                "requires_api_key": engine.requires_api_key,
                "is_free": engine.is_free,
                "is_configured": is_configured,
                "is_active": name == self.active_engine_name
            })
        return result

    def set_active(self, name):
        if name in self.engines or name is None:
            self.active_engine_name = name
            self.save_config()
            return True
        return False

    def get_active(self):
        if self.active_engine_name:
            return self.engines.get(self.active_engine_name)
        return None

    def test_engine(self, name):
        engine = self.engines.get(name)
        if engine:
            return engine.test_connection()
        return False

    def load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.active_engine_name = data.get("active_engine")
            except Exception:
                self.active_engine_name = None

    def save_config(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump({"active_engine": self.active_engine_name}, f)
        except Exception as e:
            print(f"Failed to save autodetect config: {e}")
