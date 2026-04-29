"""
wordnet_checker.py
------------------
Phase 3 — Lexical-semantic contradiction detection using the NLTK Spanish
WordNet (Open Multilingual Wordnet, language code 'spa').

Detects two types of semantic conflicts:
  1. Antonym contradiction: predicate of one claim is a direct antonym of the
     predicate of another (e.g. "aumentar" vs "disminuir").
  2. Negated-synonym contradiction: one claim negates a predicate that is
     synonymous with the other claim's predicate
     (e.g. "no confirmó" vs "aprobó").

Requires: pip install nltk
After install run once:
  import nltk; nltk.download('omw-1.4'); nltk.download('wordnet')
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Open Multilingual Wordnet language code for Spanish
_LANG = "spa"


# ---------------------------------------------------------------------------
# Lazy WordNet loader
# ---------------------------------------------------------------------------


def _get_wn():
    """Return the NLTK WordNet corpus reader (lazy import)."""
    from nltk.corpus import wordnet as wn
    return wn


# ---------------------------------------------------------------------------
# Core WordNet queries (cached for performance)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _synsets_for_lemma(lemma: str) -> Tuple:
    """Return synsets that contain the given Spanish lemma."""
    wn = _get_wn()
    return tuple(wn.synsets(lemma, lang=_LANG))


@lru_cache(maxsize=4096)
def _antonym_lemmas(lemma: str) -> frozenset:
    """
    Return all antonym lemmas (in Spanish) reachable from any synset that
    contains *lemma*.
    """
    wn = _get_wn()
    antonyms: Set[str] = set()
    for syn in _synsets_for_lemma(lemma):
        for lemma_obj in syn.lemmas(lang=_LANG):
            for ant in lemma_obj.antonyms():
                # antonyms() returns Lemma objects; get Spanish forms
                ant_syn = ant.synset()
                for es_lemma in ant_syn.lemmas(lang=_LANG):
                    antonyms.add(es_lemma.name().lower().replace("_", " "))
    return frozenset(antonyms)


@lru_cache(maxsize=4096)
def _synonym_lemmas(lemma: str) -> frozenset:
    """Return all Spanish lemmas that are synonyms of *lemma* (share a synset)."""
    wn = _get_wn()
    synonyms: Set[str] = set()
    for syn in _synsets_for_lemma(lemma):
        for es_lemma in syn.lemmas(lang=_LANG):
            name = es_lemma.name().lower().replace("_", " ")
            if name != lemma.lower():
                synonyms.add(name)
    return frozenset(synonyms)


# ---------------------------------------------------------------------------
# Negation detection (Spanish)
# ---------------------------------------------------------------------------

_ES_NEGATION_PARTICLES = {"no", "ni", "nunca", "jamás", "tampoco", "sin"}


def _is_negated(claim) -> bool:
    """
    Return True if the claim's predicate is syntactically negated.
    Uses a simple heuristic: a Spanish negation particle appears within ~25
    characters before the predicate in the sentence text.
    """
    if not claim.predicate or not claim.predicate_lemma:
        return False

    sentence_lower = claim.sentence_text.lower()
    pred_lower     = claim.predicate.lower()

    for neg in _ES_NEGATION_PARTICLES:
        pattern_idx = sentence_lower.find(neg)
        pred_idx    = sentence_lower.find(pred_lower)
        if pattern_idx != -1 and pred_idx != -1:
            gap = pred_idx - pattern_idx
            if 0 < gap < 25:
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WordNetChecker:
    """
    Detect lexical-semantic contradictions between claim pairs using the
    Open Multilingual Wordnet (Spanish).

    Checks:
      1. Direct antonymy between predicates.
      2. Negated-synonym contradiction.
    """

    def check(self, claim_a, claim_b) -> Optional[dict]:
        """
        Compare two claims for WordNet-level semantic contradiction.

        Returns a dict ``{type, explanation, evidence_a, evidence_b}``
        if a contradiction is found, else ``None``.
        """
        lemma_a = (claim_a.predicate_lemma or "").lower()
        lemma_b = (claim_b.predicate_lemma or "").lower()

        if not lemma_a or not lemma_b:
            return None

        # --- Check 1: Direct antonymy ---
        try:
            ant_a = _antonym_lemmas(lemma_a)
            if lemma_b in ant_a:
                return {
                    "type": "ANTONYM",
                    "explanation": (
                        f"Predicate '{claim_a.predicate}' (lemma: '{lemma_a}') is a "
                        f"direct antonym of '{claim_b.predicate}' (lemma: '{lemma_b}') "
                        f"according to the Spanish WordNet."
                    ),
                    "evidence_a": claim_a.predicate,
                    "evidence_b": claim_b.predicate,
                }
        except Exception as exc:
            logger.debug("Antonym lookup failed for '%s': %s", lemma_a, exc)

        # --- Check 2: Negated synonym ---
        try:
            neg_a = _is_negated(claim_a)
            neg_b = _is_negated(claim_b)

            if neg_a != neg_b:
                syn_a = _synonym_lemmas(lemma_a)
                syn_b = _synonym_lemmas(lemma_b)

                if lemma_b in syn_a or lemma_a in syn_b or lemma_a == lemma_b:
                    negated_claim  = claim_a if neg_a else claim_b
                    positive_claim = claim_b if neg_a else claim_a
                    return {
                        "type": "NEGATED_SYNONYM",
                        "explanation": (
                            f"One claim negates a predicate ('{negated_claim.predicate}') "
                            f"that is synonymous with the affirmed predicate "
                            f"('{positive_claim.predicate}') in the other claim."
                        ),
                        "evidence_a": claim_a.predicate,
                        "evidence_b": claim_b.predicate,
                    }
        except Exception as exc:
            logger.debug("Negated-synonym check failed: %s", exc)

        return None

    def batch_check(self, claims) -> List[dict]:
        """Check all claim pairs and return all WordNet contradictions found."""
        from itertools import combinations
        results = []
        for ca, cb in combinations(claims, 2):
            result = self.check(ca, cb)
            if result:
                result["claim_a_index"] = ca.sentence_index
                result["claim_b_index"] = cb.sentence_index
                result["claim_a_text"]  = ca.sentence_text
                result["claim_b_text"]  = cb.sentence_text
                results.append(result)
        return results

