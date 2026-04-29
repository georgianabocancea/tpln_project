"""
normalizer.py
-------------
Phase 2 — Normalise claim attributes to canonical forms.

  • Numeric: value + canonical SI/domain unit
  • Temporal: ISO 8601 date string (YYYY, YYYY-MM, or YYYY-MM-DD)

These canonical representations are what the deterministic contradiction
detector (Phase 3) compares.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

from src.pipeline.claim_extractor import NumericAttribute, TemporalAttribute

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numeric normalisation
# ---------------------------------------------------------------------------

# Maps Romanian unit strings to a canonical English label and a multiplier
# relative to the base unit.
_UNIT_TABLE = {
    # Currency (no multiplier — kept as absolute)
    "lei": ("RON", 1),
    "leu": ("RON", 1),
    "euro": ("EUR", 1),
    "dolari": ("USD", 1),
    "dolar": ("USD", 1),
    # Percentage
    "%": ("%", 1),
    "procente": ("%", 1),
    "procent": ("%", 1),
    # Large number multipliers
    "mii": ("units", 1_000),
    "mie": ("units", 1_000),
    "milioane": ("units", 1_000_000),
    "milion": ("units", 1_000_000),
    "miliarde": ("units", 1_000_000_000),
    "miliard": ("units", 1_000_000_000),
    "sute": ("units", 100),
    # Distance
    "km": ("km", 1),
    "m": ("m", 1),
    "cm": ("cm", 0.01),
    # Mass
    "kg": ("kg", 1),
    "g": ("g", 0.001),
    "tone": ("t", 1_000),
    "tona": ("t", 1_000),
    # Volume
    "litri": ("L", 1),
    "litru": ("L", 1),
    # Power
    "mw": ("MW", 1),
    "gw": ("GW", 1_000),
    "kw": ("kW", 0.001),
    # Time durations
    "ani": ("years", 1),
    "an": ("years", 1),
    "luni": ("months", 1),
    "luna": ("months", 1),
    "zile": ("days", 1),
    "zi": ("days", 1),
    "ore": ("hours", 1),
    "ora": ("hours", 1),
    "minute": ("minutes", 1),
    "minut": ("minutes", 1),
    "secunde": ("seconds", 1),
    "secunda": ("seconds", 1),
}


def normalize_numeric(attr: NumericAttribute) -> Tuple[float, Optional[str]]:
    """
    Return (canonical_value, canonical_unit).

    For example:
      NumericAttribute("3 milioane", 3.0, "milioane") → (3_000_000.0, "units")
      NumericAttribute("2,5%", 2.5, "%") → (2.5, "%")
    """
    unit_key = (attr.unit or "").strip().lower()
    if unit_key in _UNIT_TABLE:
        canonical_unit, multiplier = _UNIT_TABLE[unit_key]
        return attr.value * multiplier, canonical_unit
    return attr.value, attr.unit  # unknown unit — return as-is


# ---------------------------------------------------------------------------
# Temporal normalisation
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "ianuarie": "01", "februarie": "02", "martie": "03",
    "aprilie": "04", "mai": "05", "iunie": "06",
    "iulie": "07", "august": "08", "septembrie": "09",
    "octombrie": "10", "noiembrie": "11", "decembrie": "12",
}

_QUARTER_MAP = {
    "primul trimestru": "Q1",
    "al doilea trimestru": "Q2",
    "al treilea trimestru": "Q3",
    "al patrulea trimestru": "Q4",
}

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_FULL_DATE_RE = re.compile(
    rf"(\d{{1,2}})\s+({'|'.join(_MONTH_MAP.keys())})\s+((?:19|20)\d{{2}})",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    rf"({'|'.join(_MONTH_MAP.keys())})\s+((?:19|20)\d{{2}})",
    re.IGNORECASE,
)


def normalize_temporal(attr: TemporalAttribute) -> Optional[str]:
    """
    Return an ISO 8601 string for the temporal expression, or None if
    normalisation is not possible.

    Examples:
      "3 martie 2024" → "2024-03-03"
      "martie 2024"   → "2024-03"
      "2024"          → "2024"
      "primul trimestru" → "Q1"  (non-ISO but canonical for quarters)
    """
    raw = attr.raw_text.strip().lower()

    # Full date: DD Month YYYY
    m = _FULL_DATE_RE.search(raw)
    if m:
        day = m.group(1).zfill(2)
        month = _MONTH_MAP[m.group(2).lower()]
        year = m.group(3)
        return f"{year}-{month}-{day}"

    # Month + Year
    m = _MONTH_YEAR_RE.search(raw)
    if m:
        month = _MONTH_MAP[m.group(1).lower()]
        year = m.group(2)
        return f"{year}-{month}"

    # Bare year
    m = _YEAR_RE.search(raw)
    if m:
        return m.group(1)

    # Quarter expressions
    for ro_quarter, iso_q in _QUARTER_MAP.items():
        if ro_quarter in raw:
            # Try to find accompanying year
            year_m = _YEAR_RE.search(raw)
            if year_m:
                return f"{year_m.group(1)}-{iso_q}"
            return iso_q

    return None  # could not normalise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Normalizer:
    """
    Apply canonical normalisation to all attributes in a list of Claims.

    This does NOT mutate the original Claim objects — it returns a dict
    of normalised values keyed to each claim.

    Usage
    -----
    >>> normalizer = Normalizer()
    >>> norm_data = normalizer.normalize_claims(claims)
    >>> for claim, data in zip(claims, norm_data):
    ...     print(claim, data)
    """

    def normalize_claim(self, claim) -> dict:
        """
        Return a dict with normalised numeric and temporal attributes for
        the given Claim.
        """
        norm_numerics = []
        for attr in claim.numerics:
            canon_value, canon_unit = normalize_numeric(attr)
            norm_numerics.append({
                "raw": attr.raw_text,
                "value": canon_value,
                "unit": canon_unit,
            })

        norm_temporals = []
        for attr in claim.temporals:
            iso = normalize_temporal(attr)
            if iso is not None:
                attr.normalized = iso  # mutate in-place for downstream use
            norm_temporals.append({
                "raw": attr.raw_text,
                "iso": iso,
            })

        return {
            "numerics": norm_numerics,
            "temporals": norm_temporals,
        }

    def normalize_claims(self, claims) -> list:
        """
        Normalise all claims and return a list of normalisation dicts
        (same order as input).
        """
        return [self.normalize_claim(c) for c in claims]

