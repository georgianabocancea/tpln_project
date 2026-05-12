"""
claim_extractor.py
------------------
Phase 2 — Extract structured claims from dependency-parsed Romanian sentences.

A claim is a compact claim frame centered on a predicate, enriched with:
    - subject / object spans
    - attached modifiers and other descriptive POS tags
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
    qualifiers: List[str] = field(default_factory=list)

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
            f"obj='{self.object}', quals={self.qualifiers}, nums={self.numerics}, "
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

PREDICATE_ROOT_POS = {"ADJ", "NOUN", "PROPN"}
CLAIM_MODIFIER_DEPS = {
    "det", "amod", "nummod", "nmod", "nmod:poss", "advmod", "case",
    "flat", "flat:name", "compound", "appos", "acl", "advcl", "neg",
    "mark", "fixed", "conj", "cc", "aux", "cop",
}

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


def _is_copular_auxiliary(token, tokens) -> bool:
    parent = _get_token_by_id(tokens, token.head)
    if not parent:
        return False
    return token.dep in {"cop", "aux"} and parent.pos in PREDICATE_ROOT_POS


def _predicate_candidates(tokens):
    candidates = []
    seen = set()
    for tok in tokens:
        if tok.pos in VERB_POS:
            if _is_copular_auxiliary(tok, tokens):
                continue
            if tok.index not in seen:
                candidates.append(tok)
                seen.add(tok.index)
        elif tok.head == 0 and tok.pos in PREDICATE_ROOT_POS:
            if tok.index not in seen:
                candidates.append(tok)
                seen.add(tok.index)
    return candidates


def _extract_qualifiers(tokens, predicate_idx: int, used_indices: set[int]) -> List[str]:
    qualifiers: List[str] = []
    seen = set()
    for child in tokens:
        if child.head != predicate_idx:
            continue
        if child.index in used_indices or child.dep not in CLAIM_MODIFIER_DEPS:
            continue
        span = " ".join(_collect_span(tokens, child.index))
        if span:
            key = span.lower()
            if key not in seen:
                qualifiers.append(span)
                seen.add(key)
    return qualifiers


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

def _extract_role_qualifiers(tokens, role_root_idx: int, used_indices: set) -> List[str]:
    """
    Extract qualifiers attached to a subject or object node:
    adjectival modifiers (amod), prepositional phrases (nmod + case),
    appositions (appos), numeric modifiers (nummod), adverbials (advmod).

    These enrich the claim with attributes beyond the bare head noun,
    e.g. 'creștere economică de 3%' instead of just 'creștere'.
    """
    ROLE_QUALIFIER_DEPS = {
        "amod",       # adjectival modifier: 'creștere economică'
        "nummod",     # numeric modifier:    'trei milioane'
        "nmod",       # nominal modifier:    'profitul companiei'
        "nmod:poss",  # possessive:          'profitul său'
        "appos",      # apposition:          'Ion Popescu, ministrul'
        "advmod",     # adverbial:           'mult mai mare'
        "acl",        # adjectival clause:   'suma declarată'
        "flat:name",  # flat name:           'Ion Popescu'
        "flat",       # flat compound:       compound proper nouns
    }

    qualifiers: List[str] = []
    seen_spans: set = set()

    for child in tokens:
        if child.head != role_root_idx:
            continue
        if child.dep not in ROLE_QUALIFIER_DEPS:
            continue
        # Collect the full span of this qualifier (e.g. 'de 3 milioane de euro')
        span_indices = _collect_span_indices(tokens, child.index)
        # Skip if already captured as part of subject/object
        if span_indices and span_indices[0] in used_indices:
            continue
        span_text = " ".join(
            tok.text for idx in span_indices
            for tok in [_get_token_by_id(tokens, idx)] if tok
        )
        if span_text and span_text.lower() not in seen_spans:
            qualifiers.append(span_text)
            seen_spans.add(span_text.lower())

    return qualifiers

def _extract_claims_from_sentence(sentence) -> List[Claim]:
    """Extract claim frames from a single Sentence object."""
    tokens = sentence.tokens
    claims: List[Claim] = []

    for predicate in _predicate_candidates(tokens):
        children = [t for t in tokens if t.head == predicate.index]

        subject_span: List[str] = []
        subject_lemma: Optional[str] = None
        object_span: List[str] = []
        object_lemma: Optional[str] = None
        subject_child = None
        object_child = None

        for child in children:
            if child.dep in SUBJ_DEPS:
                subject_child = child
                subject_span = _collect_span(tokens, child.index)
                subject_lemma = child.lemma
            elif child.dep in OBJ_DEPS and child.dep not in COMP_DEPS:
                object_child = child
                object_span = _collect_span(tokens, child.index)
                object_lemma = child.lemma

        if predicate.head == 0 and predicate.pos in PREDICATE_ROOT_POS and not subject_span:
            # Copular / attributive roots without a clear subject are usually
            # not useful claim frames on their own.
            continue

        if not subject_span and not object_span:
            continue

        used_indices = {predicate.index}
        if subject_child is not None:
            used_indices.update(_collect_span_indices(tokens, subject_child.index))
        if object_child is not None:
            used_indices.update(_collect_span_indices(tokens, object_child.index))

        # Qualifiers around the predicate (adverbials, complements, numeric/temporal modifiers)
        qualifiers = _extract_qualifiers(tokens, predicate.index, used_indices)
        # Also extract modifiers attached to subject/object (adjectival modifiers, nmod, prepositional phrases)
        if subject_child is not None:
            subj_quals = _extract_role_qualifiers(tokens, subject_child.index, used_indices)
            qualifiers.extend([f"subj: {q}" for q in subj_quals])
            # mark these indices as used so we don't duplicate
            for q in subj_quals:
                # collect indices for q's head (approximate)
                pass
        if object_child is not None:
            obj_quals = _extract_role_qualifiers(tokens, object_child.index, used_indices)
            qualifiers.extend([f"obj: {q}" for q in obj_quals])
            for q in obj_quals:
                pass

        claim = Claim(
            sentence_index=sentence.index,
            sentence_text=sentence.text,
            subject=" ".join(subject_span) if subject_span else None,
            subject_lemma=subject_lemma,
            predicate=predicate.text,
            predicate_lemma=predicate.lemma,
            object=" ".join(object_span) if object_span else None,
            object_lemma=object_lemma,
            qualifiers=qualifiers,
            numerics=_extract_numerics(sentence.text),
            temporals=_extract_temporals(sentence.text),
            entities=_extract_entities(tokens),
        )
        claims.append(claim)

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


