"""
claim_extractor.py
------------------
Phase 2 — Extract structured claims from dependency-parsed Romanian sentences.

A claim is a Subject–Predicate–Object (SPO) triplet, optionally enriched with:
  - numerical values (with units)
  - temporal expressions
  - named entities

Input:  List[Sentence]  (output of RomanianPreprocessor)
Output: List[Claim]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

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
    entities: Dict[str, List[str]] = field(default_factory=dict)  # label → [text, ...]

    def __repr__(self):
        return (
            f"Claim(subj='{self.subject}', pred='{self.predicate}', "
            f"obj='{self.object}', nums={self.numerics}, "
            f"temps={self.temporals}, ents={self.entities})"
        )


# ---------------------------------------------------------------------------
# Dependency relation labels used by Romanian parsers
# (spaCy ro_core_news_lg uses UD labels; Teprolin uses similar UD labels)
# ---------------------------------------------------------------------------

# Labels that mark a nominal/clausal subject
SUBJ_DEPS = {"nsubj", "nsubj:pass", "csubj", "expl:subj"}

# Labels that mark an object
OBJ_DEPS = {"obj", "iobj", "obl", "obl:agent", "xcomp"}

# Labels that mark clausal complements (treated as embedded predicates)
COMP_DEPS = {"xcomp", "ccomp"}

# POS tags for verbs
VERB_POS = {"VERB", "AUX"}

# POS tags for nouns / pronouns
NOUN_POS = {"NOUN", "PROPN", "PRON"}

# ---------------------------------------------------------------------------
# Numeric patterns (Romanian)
# ---------------------------------------------------------------------------

# Matches numbers like "3", "3.5", "3,5", "1.000.000", optionally with a %
_NUMBER_RE = re.compile(
    r"""
    \b
    (\d{1,3}(?:[.,]\d{3})*   # optional thousands separator
    (?:[.,]\d+)?              # optional decimal
    |\d+)
    \s*(%|
        milioane?|miliard(?:e|oane)?|mii|sute|
        procente?|lei?|euro?|dolari?|
        km|m|cm|kg|g|tone?|litri?|MW|GW|kW|
        ani?|luni?|zile?|ore?|minute?|secunde?
    )?
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Temporal expression patterns (simple Romanian heuristics)
# A more robust approach uses HeidelTime; this serves as a bootstrap.
# ---------------------------------------------------------------------------

_MONTHS = (
    "ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|"
    "septembrie|octombrie|noiembrie|decembrie"
)

_TEMPORAL_RE = re.compile(
    rf"""
    \b(
        (?:\d{{1,2}}\s+)?(?:{_MONTHS})(?:\s+\d{{4}})?   # e.g. "3 martie 2024"
        |\d{{4}}                                           # bare year
        |(?:primul?|al\s+doilea?|al\s+treilea?|al\s+patrulea?)
         \s+trimestru                                      # "primul trimestru"
        |(?:luni?|marți|miercuri|joi|vineri|sâmbătă|duminică)  # day names
        |ieri|azi|astăzi|mâine|acum                        # relative
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_token_by_id(tokens, idx):
    """Return the token with the given 1-based index, or None."""
    for t in tokens:
        if t.index == idx:
            return t
    return None


def _collect_span_indices(tokens, root_idx: int, visited=None) -> List[int]:
    """
    Recursively collect token indices for the noun phrase rooted at root_idx.
    Returns a sorted list of 1-based token indices.
    """
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
    """
    Collect the full noun phrase rooted at root_idx by following dependents
    that are modifiers, determiners, etc. Returns token texts in order.
    """
    indices = _collect_span_indices(tokens, root_idx, visited)
    texts = []
    for idx in indices:
        tok = _get_token_by_id(tokens, idx)
        if tok:
            texts.append(tok.text)
    return texts


def _extract_numerics(text: str) -> List[NumericAttribute]:
    """Extract all numeric expressions from a raw text string."""
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
    """Extract temporal expressions from a raw text string (simple heuristics)."""
    temporals = []
    for m in _TEMPORAL_RE.finditer(text):
        raw = m.group(0).strip()
        temporals.append(TemporalAttribute(raw_text=raw))
    return temporals


def _extract_entities(tokens) -> Dict[str, List[str]]:
    """Group named entities by their NER label (from token BIO tags)."""
    entities: Dict[str, List[str]] = {}
    current_label = None
    current_parts = []

    def flush():
        if current_label and current_parts:
            label = current_label.split("-", 1)[-1]  # strip B-/I-
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
            # Non-BIO NE tag (e.g. Teprolin uses full label without prefix)
            flush()
            current_label = f"B-{tag}"
            current_parts = [tok.text]

    flush()
    return entities


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def _extract_claims_from_sentence(sentence) -> List[Claim]:
    """
    Extract SPO claims from a single Sentence object.

    Strategy:
      1. Find all VERB tokens (potential predicates).
      2. For each verb, find its nsubj (subject) and obj/obl (object) dependents.
      3. Collect NP spans for subject and object.
      4. Attach numeric, temporal, and NE attributes to the claim.
    """
    from src.pipeline.preprocessor import Sentence, Token  # avoid circular import

    tokens = sentence.tokens
    claims: List[Claim] = []

    # Index tokens by their id for fast lookup
    tok_by_id = {t.index: t for t in tokens}

    # Find all verb tokens
    verb_tokens = [t for t in tokens if t.pos in VERB_POS]

    for verb in verb_tokens:
        # Find direct children
        children = [t for t in tokens if t.head == verb.index]

        subject_span = []
        subject_lemma = None
        object_span = []
        object_lemma = None

        for child in children:
            if child.dep in SUBJ_DEPS:
                span_texts = _collect_span(tokens, child.index)
                subject_span = span_texts
                subject_lemma = child.lemma
            elif child.dep in OBJ_DEPS and child.dep not in COMP_DEPS:
                span_texts = _collect_span(tokens, child.index)
                object_span = span_texts
                object_lemma = child.lemma

        # Only emit a claim if we have at least a subject or an object
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

    # If no verb was found, emit a minimal claim just for NE/numeric coverage
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
    Extract structured claims from preprocessed Romanian sentences.

    Usage
    -----
    >>> from src.pipeline.preprocessor import RomanianPreprocessor
    >>> from src.pipeline.claim_extractor import ClaimExtractor
    >>> prep = RomanianPreprocessor()
    >>> extractor = ClaimExtractor()
    >>> sentences = prep.process("Guvernul a anunțat că economia a crescut cu 3%.")
    >>> claims = extractor.extract(sentences)
    """

    def extract(self, sentences) -> List[Claim]:
        """
        Extract claims from a list of Sentence objects.

        Parameters
        ----------
        sentences : List[Sentence]

        Returns
        -------
        List[Claim]
        """
        all_claims: List[Claim] = []
        for sent in sentences:
            claims = _extract_claims_from_sentence(sent)
            all_claims.extend(claims)
        return all_claims

    def extract_from_text(self, text: str, preprocessor=None) -> List[Claim]:
        """
        Convenience method: preprocess and extract in one call.

        Parameters
        ----------
        text : str
            Raw Romanian article text.
        preprocessor : RomanianPreprocessor, optional
            If None, a new one is created with default (auto) backend.
        """
        if preprocessor is None:
            from src.pipeline.preprocessor import RomanianPreprocessor
            preprocessor = RomanianPreprocessor()
        sentences = preprocessor.process(text)
        return self.extract(sentences)


