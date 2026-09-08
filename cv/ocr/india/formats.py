import re

from cv.ocr.india.models import IndiaPlateFormat, IndiaPlateValidation, PrefixStatus
from cv.ocr.india.prefixes import CURRENT_PREFIXES, LEGACY_PREFIXES

_BH = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$")
_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")
_STANDARD_LIKE = re.compile(r"^[A-Z]{2}[0-9OIQ]{1,2}[A-Z]{1,3}[0-9]{1,4}$")
_SPECIAL = re.compile(r"^(?:[0-9]{2}[A-Z][0-9]|[A-Z]{1,3}[0-9]{5,})")


def classify_india_plate(text: str) -> IndiaPlateValidation:
    if _BH.fullmatch(text):
        return IndiaPlateValidation(text, IndiaPlateFormat.BH_SERIES, None,
                                    PrefixStatus.NOT_APPLICABLE, True, None)
    if _STANDARD.fullmatch(text) or _STANDARD_LIKE.fullmatch(text):
        prefix = text[:2]
        structurally_valid = bool(_STANDARD.fullmatch(text))
        if prefix in CURRENT_PREFIXES:
            status, valid = PrefixStatus.VALID_CURRENT, True
        elif prefix in LEGACY_PREFIXES:
            status, valid = PrefixStatus.VALID_LEGACY, True
        else:
            status, valid = PrefixStatus.INVALID, False
        reasons = [] if valid else ["INVALID_STATE_PREFIX"]
        if not structurally_valid:
            reasons.append("NON_DIGIT_REGISTERING_AUTHORITY")
        return IndiaPlateValidation(text, IndiaPlateFormat.STANDARD_STATE, prefix,
                                    status, structurally_valid, valid, tuple(reasons))
    if _SPECIAL.match(text):
        return IndiaPlateValidation(text, IndiaPlateFormat.SPECIAL, None,
                                    PrefixStatus.NOT_APPLICABLE, True, None)
    return IndiaPlateValidation(text, IndiaPlateFormat.UNKNOWN, None,
                                PrefixStatus.UNKNOWN, False, None, ("UNCLASSIFIED_FORMAT",))
