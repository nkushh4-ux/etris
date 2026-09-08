"""OCR recognition, evidence conversion, and multi-frame fusion."""

from cv.ocr.engine import OCREngine
from cv.ocr.models import OCRResult, OCRResultStatus

__all__ = ("OCREngine", "OCRResult", "OCRResultStatus")
