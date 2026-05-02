"""
contradiction_detector.py
--------------------------
Phase 3 — Detect contradictions between claims extracted from a single Spanish
news article.

Two detection layers:

  Layer 1 — Deterministic rules (hard conflicts):
    • Numeric contradiction: same subject+predicate, different normalised values
    • Temporal contradiction: same subject+predicate, different normalised dates
    • Entity contradiction: same role, mutually exclusive NE values

  Layer 1b — WordNet antonymy / negated-synonym (NLTK Spanish WordNet)

  Layer 2 — NLI-based soft detection (multilingual XLM-RoBERTa, optional):
    • Reuses the same NLI module as the Romanian pipeline (multilingual model)

Each detected contradiction is a ContradictionAlert with full explainability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CONTRADICTION_TYPES = {
    "NUMERIC":    "Numeric conflict between normalised values",
    "TEMPORAL":   "Temporal conflict between normalised dates",
    "ENTITY":     "Entity conflict (incompatible named entities in same role)",
    "LINGUISTIC": "Soft linguistic contradiction detected by NLI / WordNet",
}


@dataclass
class ContradictionAlert:
    claim_a_index: int
    claim_b_index: int
    claim_a_text: str
    claim_b_text: str
    contradiction_type: str     # one of CONTRADICTION_TYPES keys
    explanation: str
    confidence: float = 1.0
    evidence_a: str = ""
    evidence_b: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_a_sentence": self.claim_a_index,
            "claim_b_sentence": self.claim_b_index,
            "claim_a_text":     self.claim_a_text,
            "claim_b_text":     self.claim_b_text,
            "type":             self.contradiction_type,
            "explanation":      self.explanation,
            "confidence":       self.confidence,
            "evidence_a":       self.evidence_a,
            "evidence_b":       self.evidence_b,
        }


# ---------------------------------------------------------------------------
# Helper: predicate / subject similarity
# ---------------------------------------------------------------------------


def _same_predicate(a, b) -> bool:
    return (
        a.predicate_lemma is not None
        and b.predicate_lemma is not None
        and a.predicate_lemma.lower() == b.predicate_lemma.lower()
    )


def _similar_subject(a, b) -> bool:
    if a.subject_lemma and b.subject_lemma:
        return a.subject_lemma.lower() == b.subject_lemma.lower()
    return False


def _shared_topic_word(a, b) -> Optional[str]:
    lemmas_a = {l.lower() for l in [a.subject_lemma, a.object_lemma] if l}
    lemmas_b = {l.lower() for l in [b.subject_lemma, b.object_lemma] if l}
    shared = lemmas_a & lemmas_b
    return next(iter(shared), None) if shared else None


def _claims_are_related(a, b, require_evidence: bool = False) -> bool:
    if _same_predicate(a, b) and _similar_subject(a, b):
        return True
    
    # FIX: same subject is enough to consider the claims are related
    if _similar_subject(a, b):
        if require_evidence:
            return bool(a.numerics or a.temporals) and bool(b.numerics or b.temporals)
        return True
    
    topic = _shared_topic_word(a, b)
    if topic:
        if require_evidence:
            return bool(a.numerics or a.temporals) and bool(b.numerics or b.temporals)
        return True
    return False


# ---------------------------------------------------------------------------
# Layer 1 — Deterministic rules
# ---------------------------------------------------------------------------

_NUMERIC_REL_TOLERANCE = 0.01   # 1%


def _check_numeric_contradiction(
    claim_a, claim_b, norm_a: dict, norm_b: dict
) -> Optional[ContradictionAlert]:
    if not norm_a["numerics"] or not norm_b["numerics"]:
        return None
    if not _claims_are_related(claim_a, claim_b, require_evidence=True):
        return None

    for na in norm_a["numerics"]:
        for nb in norm_b["numerics"]:
            if na["unit"] != nb["unit"]:
                continue
            va, vb = na["value"], nb["value"]
            if va == 0 and vb == 0:
                continue
            denom   = max(abs(va), abs(vb), 1e-9)
            rel_diff = abs(va - vb) / denom
            if rel_diff > _NUMERIC_REL_TOLERANCE:
                return ContradictionAlert(
                    claim_a_index=claim_a.sentence_index,
                    claim_b_index=claim_b.sentence_index,
                    claim_a_text=claim_a.sentence_text,
                    claim_b_text=claim_b.sentence_text,
                    contradiction_type="NUMERIC",
                    explanation=(
                        f"The same predicate '{claim_a.predicate_lemma}' is associated "
                        f"with conflicting numeric values: {na['value']} {na['unit']} "
                        f"vs {nb['value']} {nb['unit']} "
                        f"(relative difference: {rel_diff * 100:.1f}%)."
                    ),
                    confidence=1.0,
                    evidence_a=na["raw"],
                    evidence_b=nb["raw"],
                )
    return None


def _check_temporal_contradiction(
    claim_a, claim_b, norm_a: dict, norm_b: dict
) -> Optional[ContradictionAlert]:
    if not norm_a["temporals"] or not norm_b["temporals"]:
        return None
    if not _claims_are_related(claim_a, claim_b, require_evidence=True):
        return None

    iso_a = [t["iso"] for t in norm_a["temporals"] if t["iso"]]
    iso_b = [t["iso"] for t in norm_b["temporals"] if t["iso"]]
    if not iso_a or not iso_b:
        return None

    for ta in iso_a:
        for tb in iso_b:
            if ta != tb and not ta.startswith(tb) and not tb.startswith(ta):
                return ContradictionAlert(
                    claim_a_index=claim_a.sentence_index,
                    claim_b_index=claim_b.sentence_index,
                    claim_a_text=claim_a.sentence_text,
                    claim_b_text=claim_b.sentence_text,
                    contradiction_type="TEMPORAL",
                    explanation=(
                        f"Conflicting temporal expressions for the same event "
                        f"(predicate '{claim_a.predicate_lemma}'): '{ta}' vs '{tb}'."
                    ),
                    confidence=1.0,
                    evidence_a=norm_a["temporals"][0]["raw"],
                    evidence_b=norm_b["temporals"][0]["raw"],
                )
    return None


def _check_entity_contradiction(
    claim_a, claim_b
) -> Optional[ContradictionAlert]:
    if not claim_a.entities or not claim_b.entities:
        return None
    if not _same_predicate(claim_a, claim_b):
        return None

    for label in claim_a.entities:
        if label not in claim_b.entities:
            continue
        ents_a = {e.lower() for e in claim_a.entities[label]}
        ents_b = {e.lower() for e in claim_b.entities[label]}
        if ents_a and ents_b and ents_a.isdisjoint(ents_b):
            return ContradictionAlert(
                claim_a_index=claim_a.sentence_index,
                claim_b_index=claim_b.sentence_index,
                claim_a_text=claim_a.sentence_text,
                claim_b_text=claim_b.sentence_text,
                contradiction_type="ENTITY",
                explanation=(
                    f"Conflicting {label} entities associated with the same predicate "
                    f"'{claim_a.predicate_lemma}': "
                    f"{sorted(ents_a)} vs {sorted(ents_b)}."
                ),
                confidence=0.8,
                evidence_a=", ".join(sorted(ents_a)),
                evidence_b=", ".join(sorted(ents_b)),
            )
    return None


# ---------------------------------------------------------------------------
# Layer 1b — WordNet antonymy (Spanish via NLTK)
# ---------------------------------------------------------------------------


def _check_wordnet_contradiction(claim_a, claim_b) -> Optional[ContradictionAlert]:
    try:
        from src.spanish_pipeline.wordnet_checker import WordNetChecker
        checker = WordNetChecker()
        result = checker.check(claim_a, claim_b)
        if result:
            return ContradictionAlert(
                claim_a_index=claim_a.sentence_index,
                claim_b_index=claim_b.sentence_index,
                claim_a_text=claim_a.sentence_text,
                claim_b_text=claim_b.sentence_text,
                contradiction_type="LINGUISTIC",
                explanation=result["explanation"],
                confidence=0.9,
                evidence_a=result["evidence_a"],
                evidence_b=result["evidence_b"],
            )
    except Exception as exc:
        logger.debug("WordNet check failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Layer 2 — NLI-based soft detection  (reuses Romanian nli_module)
# ---------------------------------------------------------------------------


def _check_nli_contradiction(
    claim_a, claim_b, threshold: float = 0.65
) -> Optional[ContradictionAlert]:
    """
    Delegates to the shared multilingual NLI module (XLM-RoBERTa).
    The model supports Spanish natively.
    """
    try:
        from src.pipeline.nli_module import get_default_scorer
        scorer = get_default_scorer(threshold=threshold)
        is_contra, score = scorer.is_contradiction(
            claim_a.sentence_text, claim_b.sentence_text
        )
        if is_contra:
            return ContradictionAlert(
                claim_a_index=claim_a.sentence_index,
                claim_b_index=claim_b.sentence_index,
                claim_a_text=claim_a.sentence_text,
                claim_b_text=claim_b.sentence_text,
                contradiction_type="LINGUISTIC",
                explanation=(
                    f"Multilingual NLI model detected a linguistic contradiction "
                    f"(confidence: {score:.2f})."
                ),
                confidence=score,
                evidence_a=claim_a.sentence_text,
                evidence_b=claim_b.sentence_text,
            )
    except Exception as exc:
        logger.warning("NLI check failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ContradictionDetector:
    """
    Detect contradictions between all pairs of claims in a Spanish document.

    Parameters
    ----------
    use_wordnet : bool
        Enable NLTK Spanish WordNet antonymy / negated-synonym detection.
    use_nli : bool
        Enable multilingual NLI soft-contradiction detection (slower).
    nli_threshold : float
        Minimum NLI contradiction probability to emit an alert (0–1).
    """

    def __init__(
        self,
        use_wordnet: bool = True,
        use_nli: bool = False,
        nli_threshold: float = 0.65,
    ):
        self.use_wordnet   = use_wordnet
        self.use_nli       = use_nli
        self.nli_threshold = nli_threshold

    def detect(self, claims, norm_data: list) -> List[ContradictionAlert]:
        """
        Compare all pairs of claims and return detected contradictions.

        Parameters
        ----------
        claims : List[Claim]
        norm_data : List[dict]
            Normalised attributes per claim (output of Normalizer).

        Returns
        -------
        List[ContradictionAlert]
        """
        alerts: List[ContradictionAlert] = []

        for (i, claim_a), (j, claim_b) in combinations(enumerate(claims), 2):
            norm_a = norm_data[i]
            norm_b = norm_data[j]

            # Layer 1a: deterministic numeric / temporal / entity rules
            alert = (
                _check_numeric_contradiction(claim_a, claim_b, norm_a, norm_b)
                or _check_temporal_contradiction(claim_a, claim_b, norm_a, norm_b)
                or _check_entity_contradiction(claim_a, claim_b)
            )
            if alert:
                alerts.append(alert)
                continue

            # Layer 1b: WordNet antonymy / negated-synonym
            if self.use_wordnet:
                wn_alert = _check_wordnet_contradiction(claim_a, claim_b)
                if wn_alert:
                    alerts.append(wn_alert)
                    continue

            # Layer 2: NLI soft detection
            if self.use_nli:
                nli_alert = _check_nli_contradiction(
                    claim_a, claim_b, self.nli_threshold
                )
                if nli_alert:
                    alerts.append(nli_alert)

        return alerts


