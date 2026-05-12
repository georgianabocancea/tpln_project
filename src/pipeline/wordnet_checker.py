"""
wordnet_checker.py
------------------
Phase 3 — Lexical-semantic contradiction detection using the Romanian WordNet (RoWordNet).

Detects two types of semantic conflicts:
  1. Antonym contradiction: the predicate of one claim is a direct antonym
     of the predicate of another (e.g. "crește" vs "scade").
  2. Negated-synonym contradiction: one claim negates a predicate that is
     synonymous with the other claim's predicate
     (e.g. "nu a confirmat" vs "a aprobat").

Requires: pip install rowordnet
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Set, List, Tuple

from pyarrow.acero import exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load RoWordNet (lazy, singleton)
# ---------------------------------------------------------------------------

_rwn = None


def _get_rwn():
    global _rwn
    if _rwn is None:
        import rowordnet
        logger.info("Loading RoWordNet…")
        _rwn = rowordnet.RoWordNet()
        logger.info("RoWordNet loaded — %d synsets.", len(list(_rwn.synsets())))
    return _rwn


# ---------------------------------------------------------------------------
# Core WordNet queries (cached for performance)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _synset_ids_for_lemma(lemma: str) -> Tuple[str, ...]:
    """Return a tuple of synset IDs containing the given lemma."""
    rwn = _get_rwn()
    return tuple(rwn.synsets(literal=lemma))


@lru_cache(maxsize=4096)
def _antonym_lemmas(lemma: str) -> frozenset:
    """
    Return all antonym lemmas reachable from any synset that contains `lemma`.
    Uses the 'near_antonym' and 'antonym' relation types available in RoWordNet.

    outbound_relations(synset_id) returns List[Tuple[target_id, relation_type]].
    """
    rwn = _get_rwn()
    antonyms: Set[str] = set()
    _ANTONYM_RELS = {"antonym", "near_antonym"}

    for syn_id in _synset_ids_for_lemma(lemma):
        for target_id, rel_type in rwn.outbound_relations(syn_id):
            if rel_type in _ANTONYM_RELS:
                target_syn = rwn.synset(target_id)
                for lit in target_syn.literals:
                    antonyms.add(lit.lower())
    return frozenset(antonyms)


@lru_cache(maxsize=4096)
def _synonym_lemmas(lemma: str) -> frozenset:
    """
    Return all lemmas that are synonyms of `lemma` (share a synset).
    """
    rwn = _get_rwn()
    synonyms: Set[str] = set()
    for syn_id in _synset_ids_for_lemma(lemma):
        syn = rwn.synset(syn_id)
        for lit in syn.literals:
            if lit.lower() != lemma.lower():
                synonyms.add(lit.lower())
    return frozenset(synonyms)


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

# Romanian negation particles that can precede a predicate
_RO_NEGATION_PARTICLES = {"nu", "n-", "nici", "niciodată", "nicidecum", "fără"}


def _is_negated(claim) -> bool:
    """
    Return True if the claim's predicate is syntactically negated.
    Checks whether any token in the sentence is a negation particle
    that is a direct dependent of the predicate token.
    """
    if not claim.predicate or not claim.predicate_lemma:
        return False

    # Find the predicate token index from sentence tokens
    # We don't have direct access to the token list here — check sentence text
    # heuristic: look for "nu <predicate>" pattern in the sentence text
    sentence_lower = claim.sentence_text.lower()
    pred_lower = claim.predicate.lower()

    for neg in _RO_NEGATION_PARTICLES:
        # Negation particle appears within 3 words before the predicate
        pattern_idx = sentence_lower.find(neg)
        pred_idx = sentence_lower.find(pred_lower)
        if pattern_idx != -1 and pred_idx != -1:
            gap = pred_idx - pattern_idx
            if 0 < gap < 25:  # roughly 3 words
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WordNetChecker:
    """
    Detect lexical-semantic contradictions between claim pairs using RoWordNet.

    Checks:
      1. Direct antonymy between predicates (e.g. "crește" ↔ "scade")
      2. Negated synonym (e.g. "nu confirmă" ↔ "aprobă")
    """

    def check(self, claim_a, claim_b) -> Optional[dict]:
        """
        Compare two claims for WordNet-level semantic contradiction.

        Returns a dict with keys {type, explanation, evidence_a, evidence_b}
        if a contradiction is found, else None.
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
                        f"according to the Romanian WordNet."
                    ),
                    "evidence_a": claim_a.predicate,
                    "evidence_b": claim_b.predicate,
                }
        except Exception as exc:
            logger.debug("Antonym lookup failed for '%s': %s", lemma_a, exc)

        # --- Check 2: Negated synonym contradiction ---
        try:
            neg_a = _is_negated(claim_a)
            neg_b = _is_negated(claim_b)

            # One is negated, the other is not
            if neg_a != neg_b:
                syn_a = _synonym_lemmas(lemma_a)
                syn_b = _synonym_lemmas(lemma_b)

                if lemma_b in syn_a or lemma_a in syn_b or lemma_a == lemma_b:
            
                    # NEW: verify that the objects are semantically similar
                    # before reporting a contradiction — prevents false positives
                    # where the same predicate is negated in completely different contexts
                    obj_a = (claim_a.object_lemma or "").lower()
                    obj_b = (claim_b.object_lemma or "").lower()
            
                    # If both claims have objects, they must overlap or be similar
                    if obj_a and obj_b and obj_a != obj_b:
                        # Check if there's any lexical overlap between objects
                        words_a = set(obj_a.split())
                        words_b = set(obj_b.split())
                        if not words_a & words_b:
                            # Objects are completely different — likely different contexts
                            logger.debug(
                                "Skipping negated-synonym: objects differ ('%s' vs '%s')",
                                obj_a, obj_b
                            )
                            return None

                    negated_claim = claim_a if neg_a else claim_b
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
        """
        Check all pairs in a list of claims and return all WordNet contradictions found.
        Each result dict also includes 'claim_a_index' and 'claim_b_index'.
        """
        from itertools import combinations
        results = []
        for i, ca in enumerate(claims):
            for j, cb in enumerate(claims):
                if j <= i:
                    continue
                result = self.check(ca, cb)
                if result:
                    result["claim_a_index"] = ca.sentence_index
                    result["claim_b_index"] = cb.sentence_index
                    result["claim_a_text"] = ca.sentence_text
                    result["claim_b_text"] = cb.sentence_text
                    results.append(result)
        return results


