class OCRError(Exception): pass
class EngineNotAvailableError(OCRError): pass
class LanguageNotSupportedError(OCRError): pass
