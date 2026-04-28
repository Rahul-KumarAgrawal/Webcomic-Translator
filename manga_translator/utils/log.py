import logging
import colorama

from .generic import replace_prefix

ROOT_TAG = 'manga-translator'

class Formatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            self._style._fmt = f'{colorama.Fore.RED}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno >= logging.WARN:
            self._style._fmt = f'{colorama.Fore.YELLOW}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno == logging.DEBUG:
            self._style._fmt = '[%(name)s] %(message)s'
        else:
            self._style._fmt = '[%(name)s] %(message)s'
        return super().formatMessage(record)

class Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Allow logs from manga-translator and our project modules
        allowed_prefixes = (ROOT_TAG, 'manga_translator', 'web', 'core', 'batch_processor', 'memory', 'model', 'werkzeug', 'root', '__main__', 'manga_ocr', 'manga')
        if any(record.name.startswith(p) for p in allowed_prefixes):
            # Shorten the name if it starts with ROOT_TAG
            if record.name.startswith(ROOT_TAG):
                record.name = replace_prefix(record.name, ROOT_TAG + '.', '')
            return True
        return False

root = logging.getLogger(ROOT_TAG)

def init_logging():
    logging.basicConfig(level=logging.INFO)
    for h in logging.root.handlers:
        h.setFormatter(Formatter())
        h.addFilter(Filter())

def set_log_level(level):
    root.setLevel(level)

def get_logger(name: str):
    return root.getChild(name)

file_handlers = {}

def add_file_logger(path: str):
    if path in file_handlers:
        return
    file_handlers[path] = logging.FileHandler(path, encoding='utf8')
    logging.root.addHandler(file_handlers[path])

def remove_file_logger(path: str):
    if path in file_handlers:
        logging.root.removeHandler(file_handlers[path])
        file_handlers[path].close()
        del file_handlers[path]
