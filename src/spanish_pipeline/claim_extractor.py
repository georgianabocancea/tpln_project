"""
claim_extractor.py
------------------
Phase 2 — Extract structured claims from dependency-parsed Spanish sentences.

A claim is a Subject–Predicate–Object (SPO) triplet, optionally enriched with:
  - numerical values (with units)
  - temporal expressions
  - named entities

Input:  List[Sentence]  (output of SpanishPreprocessor)
Output: List[Claim]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NumericAttribute:
    raw_text: str
    value: float
    unit: Optional[str] = None


@dataclass
class TemporalAttribute:
    raw_text: str
    normalized: Optional[str] = None   # ISO 8601 when available


@dataclass
class Claim:
    sentence_index: int
    sentence_text: str

    # Core SPO
    subject: Optional[str] = None
    subject_lemma: Optional[str] = None
    predicate: Optional[str] = None
    predicate_lemma: Optional[str] = None
    object: Optional[str] = None
    object_lemma: Optional[str] = None

    # Attributes
    numerics: List[NumericAttribute] = field(default_factory=list)
    temporals: List[TemporalAttribute] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)   # label → [text, …]

    def __repr__(self):
        return (
            f"Claim(subj='{self.subject}', pred='{self.predicate}', "
            f"obj='{self.object}', nums={self.numerics}, "
            f"temps={self.temporals}, ents={self.entities})"
        )


# ---------------------------------------------------------------------------
# Dependency relation labels  (Universal Dependencies — same as Romanian)
# ---------------------------------------------------------------------------

SUBJ_DEPS = {"nsubj", "nsubj:pass", "csubj", "expl:subj"}
OBJ_DEPS  = {"obj", "iobj", "obl", "obl:agent", "xcomp"}
COMP_DEPS = {"xcomp", "ccomp"}

VERB_POS = {"VERB", "AUX"}
NOUN_POS = {"NOUN", "PROPN", "PRON"}


# ---------------------------------------------------------------------------
# Numeric patterns (Spanish)
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    \b
    (\d{1,3}(?:[.,]\d{3})*   # optional thousands separator
    (?:[.,]\d+)?              # optional decimal
    |\d+)
    \s*(%|
        millones?|millardo?s?|miles?|cientos?|
        por\s+ciento|
        euros?|dólares?|pesos?|reales?|
        km|m|cm|kg|g|toneladas?|litros?|MW|GW|kW|
        años?|meses?|días?|horas?|minutos?|segundos?
    )?
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Temporal expression patterns (Spanish heuristics)
# ---------------------------------------------------------------------------

_MONTHS_ES = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    "septiembre|octubre|noviembre|diciembre"
)

_TEMPORAL_RE = re.compile(
    rf"""
    \b(
        (?:\d{{1,2}}\s+de\s+)?(?:{_MONTHS_ES})(?:\s+de\s+\d{{4}})?  # "3 de marzo de 2024"
        |\d{{4}}                                                        # bare year
        |(?:primer|segundo|tercer|cuarto)\s+trimestre                   # "primer trimestre"
        |(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)       # day names
        |ayer|hoy|mañana|ahora                                          # relative
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers  (identical logic to Romanian claim_extractor)
# ---------------------------------------------------------------------------


def _get_token_by_id(tokens, idx):
    for t in tokens:
        if t.index == idx:
            return t
    return None


def _collect_span_indices(tokens, root_idx: int, visited=None) -> List[int]:
    if visited is None:
        visited = set()
    if root_idx in visited:
        return []
    visited.add(root_idx)

    modifier_deps = {
        "det", "amod", "nummod", "nmod", "nmod:poss",
        "advmod", "case", "flat", "flat:name", "compound",
        "appos", "conj",
    }
    span_indices = [root_idx]
    for t in tokens:
        if t.head == root_idx and t.dep in modifier_deps and t.index not in visited:
            span_indices.extend(_collect_span_indices(tokens, t.index, visited))
    return sorted(set(span_indices))


def _collect_span(tokens, root_idx: int, visited=None) -> List[str]:
    indices = _collect_span_indices(tokens, root_idx, visited)
    texts = []
    for idx in indices:
        tok = _get_token_by_id(tokens, idx)
        if tok:
            texts.append(tok.text)
    return texts


def _extract_numerics(text: str) -> List[NumericAttribute]:
    numerics = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).strip()
        num_str = m.group(1).replace(".", "").replace(",", ".")
        unit = m.group(2)
        try:
            value = float(num_str)
            numerics.append(NumericAttribute(raw_text=raw, value=value, unit=unit))
        except ValueError:
            pass
    return numerics


def _extract_temporals(text: str) -> List[TemporalAttribute]:
    temporals = []
    for m in _TEMPORAL_RE.finditer(text):
        raw = m.group(0).strip()
        temporals.append(TemporalAttribute(raw_text=raw))
    return temporals


def _extract_entities(tokens) -> Dict[str, List[str]]:
    """Group named entities by their NER label (from token BIO tags)."""
    entities: Dict[str, List[str]] = {}
    current_label = None
    current_parts: List[str] = []

    def flush():
        if current_label and current_parts:
            label = current_label.split("-", 1)[-1]
            entities.setdefault(label, []).append(" ".join(current_parts))

    for tok in tokens:
        tag = tok.ner
        if tag == "O":
            flush()
            current_label = None
            current_parts = []
        elif tag.startswith("B-"):
            flush()
            current_label = tag
            current_parts = [tok.text]
        elif tag.startswith("I-"):
            current_parts.append(tok.text)
        else:
            flush()
            current_label = f"B-{tag}"
            current_parts = [tok.text]

    flush()
    return entities


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def _extract_claims_from_sentence(sentence) -> List[Claim]:
    tokens = sentence.tokens
    claims: List[Claim] = []

    verb_tokens = [t for t in tokens if t.pos in VERB_POS]

    for verb in verb_tokens:
        children = [t for t in tokens if t.head == verb.index]

        subject_span: List[str] = []
        subject_lemma: Optional[str] = None
        object_span: List[str] = []
        object_lemma: Optional[str] = None

        for child in children:
            if child.dep in SUBJ_DEPS:
                subject_span = _collect_span(tokens, child.index)
                subject_lemma = child.lemma
            elif child.dep in OBJ_DEPS and child.dep not in COMP_DEPS:
                object_span = _collect_span(tokens, child.index)
                object_lemma = child.lemma

        if not subject_span and not object_span:
            continue

        claim = Claim(
            sentence_index=sentence.index,
            sentence_text=sentence.text,
            subject=" ".join(subject_span) if subject_span else None,
            subject_lemma=subject_lemma,
            predicate=verb.text,
            predicate_lemma=verb.lemma,
            object=" ".join(object_span) if object_span else None,
            object_lemma=object_lemma,
            numerics=_extract_numerics(sentence.text),
            temporals=_extract_temporals(sentence.text),
            entities=_extract_entities(tokens),
        )
        claims.append(claim)

    # Fallback: emit a minimal claim when only NE/numeric/temporal info exists
    if not claims:
        entities = _extract_entities(tokens)
        numerics = _extract_numerics(sentence.text)
        temporals = _extract_temporals(sentence.text)
        if entities or numerics or temporals:
            claims.append(Claim(
                sentence_index=sentence.index,
                sentence_text=sentence.text,
                numerics=numerics,
                temporals=temporals,
                entities=entities,
            ))

    return claims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ClaimExtractor:
    """
    Extract structured claims from preprocessed Spanish sentences.

    Usage
    -----
    >>> from src.spanish_pipeline.preprocessor import SpanishPreprocessor
    >>> from src.spanish_pipeline.claim_extractor import ClaimExtractor
    >>> prep = SpanishPreprocessor()
    >>> extractor = ClaimExtractor()
    >>> sentences = prep.process("El gobierno anunció que la economía creció un 3%.")
    >>> claims = extractor.extract(sentences)
    """

    def extract(self, sentences) -> List[Claim]:
        """Extract claims from a list of Sentence objects."""
        all_claims: List[Claim] = []
        for sent in sentences:
            all_claims.extend(_extract_claims_from_sentence(sent))
        return all_claims

    def extract_from_text(self, text: str, preprocessor=None) -> List[Claim]:
        """
        Convenience method: preprocess and extract in one call.

        Parameters
        ----------
        text : str
            Raw Spanish article text.
        preprocessor : SpanishPreprocessor, optional
            If None, a new one is created with the default (auto) backend.
        """
        if preprocessor is None:
            from src.spanish_pipeline.preprocessor import SpanishPreprocessor
            preprocessor = SpanishPreprocessor()
        sentences = preprocessor.process(text)
        return self.extract(sentences)

