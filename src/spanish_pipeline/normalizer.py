"""
normalizer.py
-------------
Phase 2 — Normalise claim attributes to canonical forms for Spanish text.

  • Numeric: value + canonical SI / domain unit
  • Temporal: ISO 8601 date string (YYYY, YYYY-MM, or YYYY-MM-DD)
"""

from __future__ import annotations

import re
import logging
from typing import Optional, Tuple

from src.spanish_pipeline.claim_extractor import NumericAttribute, TemporalAttribute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numeric normalisation
# ---------------------------------------------------------------------------

# Maps Spanish unit strings → (canonical_label, multiplier_to_base_unit)
_UNIT_TABLE = {
    # Currency
    "euro": ("EUR", 1),
    "euros": ("EUR", 1),
    "dólar": ("USD", 1),
    "dólares": ("USD", 1),
    "peso": ("MXN", 1),
    "pesos": ("MXN", 1),
    "real": ("BRL", 1),
    "reales": ("BRL", 1),
    # Percentage
    "%": ("%", 1),
    "por ciento": ("%", 1),
    # Large number multipliers
    "miles": ("units", 1_000),
    "mil": ("units", 1_000),
    "cientos": ("units", 100),
    "millones": ("units", 1_000_000),
    "millón": ("units", 1_000_000),
    "millardo": ("units", 1_000_000_000),
    "millardos": ("units", 1_000_000_000),
    # Distance
    "km": ("km", 1),
    "m": ("m", 1),
    "cm": ("cm", 0.01),
    # Mass
    "kg": ("kg", 1),
    "g": ("g", 0.001),
    "tonelada": ("t", 1_000),
    "toneladas": ("t", 1_000),
    # Volume
    "litro": ("L", 1),
    "litros": ("L", 1),
    # Power
    "mw": ("MW", 1),
    "gw": ("GW", 1_000),
    "kw": ("kW", 0.001),
    # Time durations
    "año": ("years", 1),
    "años": ("years", 1),
    "mes": ("months", 1),
    "meses": ("months", 1),
    "día": ("days", 1),
    "días": ("days", 1),
    "hora": ("hours", 1),
    "horas": ("hours", 1),
    "minuto": ("minutes", 1),
    "minutos": ("minutes", 1),
    "segundo": ("seconds", 1),
    "segundos": ("seconds", 1),
}


def normalize_numeric(attr: NumericAttribute) -> Tuple[float, Optional[str]]:
    """
    Return (canonical_value, canonical_unit).

    Examples
    --------
    NumericAttribute("3 millones", 3.0, "millones") → (3_000_000.0, "units")
    NumericAttribute("2,5%", 2.5, "%") → (2.5, "%")
    """
    unit_key = (attr.unit or "").strip().lower()
    if unit_key in _UNIT_TABLE:
        canonical_unit, multiplier = _UNIT_TABLE[unit_key]
        return attr.value * multiplier, canonical_unit
    return attr.value, attr.unit


# ---------------------------------------------------------------------------
# Temporal normalisation
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "enero": "01", "febrero": "02", "marzo": "03",
    "abril": "04", "mayo": "05", "junio": "06",
    "julio": "07", "agosto": "08", "septiembre": "09",
    "octubre": "10", "noviembre": "11", "diciembre": "12",
}

_QUARTER_MAP = {
    "primer trimestre": "Q1",
    "segundo trimestre": "Q2",
    "tercer trimestre": "Q3",
    "cuarto trimestre": "Q4",
}

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_FULL_DATE_RE = re.compile(
    rf"(\d{{1,2}})\s+de\s+({'|'.join(_MONTH_MAP.keys())})\s+de\s+((?:19|20)\d{{2}})",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    rf"({'|'.join(_MONTH_MAP.keys())})\s+de\s+((?:19|20)\d{{2}})",
    re.IGNORECASE,
)


def normalize_temporal(attr: TemporalAttribute) -> Optional[str]:
    """
    Return an ISO 8601 string for the temporal expression, or None.

    Examples
    --------
    "3 de marzo de 2024"  → "2024-03-03"
    "marzo de 2024"       → "2024-03"
    "2024"                → "2024"
    "primer trimestre"    → "Q1"
    """
    raw = attr.raw_text.strip().lower()

    # Full date: DD de Month de YYYY
    m = _FULL_DATE_RE.search(raw)
    if m:
        day   = m.group(1).zfill(2)
        month = _MONTH_MAP[m.group(2).lower()]
        year  = m.group(3)
        return f"{year}-{month}-{day}"

    # Month + Year
    m = _MONTH_YEAR_RE.search(raw)
    if m:
        month = _MONTH_MAP[m.group(1).lower()]
        year  = m.group(2)
        return f"{year}-{month}"

    # Bare year
    m = _YEAR_RE.search(raw)
    if m:
        return m.group(1)

    # Quarter expressions
    for es_quarter, iso_q in _QUARTER_MAP.items():
        if es_quarter in raw:
            year_m = _YEAR_RE.search(raw)
            if year_m:
                return f"{year_m.group(1)}-{iso_q}"
            return iso_q

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Normalizer:
    """
    Apply canonical normalisation to all attributes in a list of Claims.

    Does NOT mutate original Claim objects — returns a list of dicts
    (one per claim, same order).

    Usage
    -----
    >>> normalizer = Normalizer()
    >>> norm_data = normalizer.normalize_claims(claims)
    """

    def normalize_claim(self, claim) -> dict:
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
                attr.normalized = iso   # mutate in-place for downstream use
            norm_temporals.append({
                "raw": attr.raw_text,
                "iso": iso,
            })

        return {
            "numerics": norm_numerics,
            "temporals": norm_temporals,
        }

    def normalize_claims(self, claims) -> list:
        """Normalise all claims and return a list of normalisation dicts."""
        return [self.normalize_claim(c) for c in claims]

