"""
contradiction_detector.py
--------------------------
Phase 3 — Detect contradictions between claims extracted from a single article.

Two detection layers:

  Layer 1 — Deterministic rules (hard conflicts):
    • Numeric contradiction: same subject+predicate, different normalised values/units
    • Temporal contradiction: same subject+predicate, different normalised dates
    • Entity contradiction: same role, mutually exclusive NE values

  Layer 2 — NLI-based soft detection (Phase 4, stub for now):
    • Uses a multilingual NLI transformer (XLM-RoBERTa or mDeBERTa)
    • Fires when deterministic rules do not apply

Each detected contradiction is a ContradictionAlert with full explainability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CONTRADICTION_TYPES = {
    "NUMERIC": "Numeric conflict between normalised values",
    "TEMPORAL": "Temporal conflict between normalised dates",
    "ENTITY": "Entity conflict (incompatible named entities in same role)",
    "LINGUISTIC": "Soft linguistic contradiction detected by NLI model",
}


@dataclass
class ContradictionAlert:
    claim_a_index: int          # sentence index of first claim
    claim_b_index: int          # sentence index of second claim
    claim_a_text: str
    claim_b_text: str
    contradiction_type: str     # one of CONTRADICTION_TYPES keys
    explanation: str
    confidence: float = 1.0     # 1.0 for deterministic; model prob for NLI
    evidence_a: str = ""        # specific fragment that conflicts (claim A side)
    evidence_b: str = ""        # specific fragment that conflicts (claim B side)

    def to_dict(self) -> dict:
        return {
            "claim_a_sentence": self.claim_a_index,
            "claim_b_sentence": self.claim_b_index,
            "claim_a_text": self.claim_a_text,
            "claim_b_text": self.claim_b_text,
            "type": self.contradiction_type,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "evidence_a": self.evidence_a,
            "evidence_b": self.evidence_b,
        }


# ---------------------------------------------------------------------------
# Helper: predicate similarity (lemma-based exact match for now)
# ---------------------------------------------------------------------------

def _same_predicate(a, b) -> bool:
    """Return True if claims share the same predicate lemma."""
    return (
        a.predicate_lemma is not None
        and b.predicate_lemma is not None
        and a.predicate_lemma.lower() == b.predicate_lemma.lower()
    )


def _similar_subject(a, b) -> bool:
    """Return True if claims have overlapping subject lemmas."""
    if a.subject_lemma and b.subject_lemma:
        return a.subject_lemma.lower() == b.subject_lemma.lower()
    return False


def _shared_topic_word(a, b) -> Optional[str]:
    """
    Return the shared content lemma if both claims reference the same noun
    anywhere (subject OR object lemma). This catches cases like:
      'Compania a raportat profitul de 5M'  → subject_lemma='companie', obj='profit'
      'Profitul companiei a crescut la 12M' → subject_lemma='profit'
    where the topic 'profit' / 'companie' overlaps.
    """
    lemmas_a = set()
    lemmas_b = set()
    for attr in [a.subject_lemma, a.object_lemma]:
        if attr:
            lemmas_a.add(attr.lower())
    for attr in [b.subject_lemma, b.object_lemma]:
        if attr:
            lemmas_b.add(attr.lower())
    shared = lemmas_a & lemmas_b
    return next(iter(shared), None) if shared else None


def _claims_are_related(a, b, require_evidence: bool = False) -> bool:
    """
    Return True if two claims are likely about the same event/entity.

    Parameters
    ----------
    require_evidence : bool
        If True, shared-topic-word match is only accepted when both claims
        carry numeric or temporal evidence (prevents loose noun coincidences
        from triggering numeric/temporal comparisons).
    """
    if _same_predicate(a, b) and _similar_subject(a, b):
        return True
    topic = _shared_topic_word(a, b)
    if topic:
        if require_evidence:
            # Only accept topic-word match when both claims carry
            # numeric or temporal raw attributes (not just a coincident noun)
            a_has_evidence = bool(a.numerics or a.temporals)
            b_has_evidence = bool(b.numerics or b.temporals)
            return a_has_evidence and b_has_evidence
        return True
    return False


# ---------------------------------------------------------------------------
# Layer 1 — Deterministic contradiction rules
# ---------------------------------------------------------------------------

# Tolerance for "approximately equal" numeric comparisons
_NUMERIC_REL_TOLERANCE = 0.01   # 1%  — less than this → not a contradiction
_NUMERIC_ABS_TOLERANCE = 0.0    # for percentages etc.


def _check_numeric_contradiction(claim_a, claim_b, norm_a: dict, norm_b: dict) -> Optional[ContradictionAlert]:
    """
    Detect numeric contradictions.

    Fires when:
      - Both claims have at least one numeric attribute
      - They share the same predicate lemma AND subject lemma
      - Their canonical values differ by more than the tolerance in the same unit
    """
    if not norm_a["numerics"] or not norm_b["numerics"]:
        return None

    if not _claims_are_related(claim_a, claim_b, require_evidence=True):
        return None

    for na in norm_a["numerics"]:
        for nb in norm_b["numerics"]:
            # Only compare values with the same unit
            if na["unit"] != nb["unit"]:
                continue
            va, vb = na["value"], nb["value"]
            if va == 0 and vb == 0:
                continue
            denom = max(abs(va), abs(vb), 1e-9)
            rel_diff = abs(va - vb) / denom
            if rel_diff > _NUMERIC_REL_TOLERANCE:
                return ContradictionAlert(
                    claim_a_index=claim_a.sentence_index,
                    claim_b_index=claim_b.sentence_index,
                    claim_a_text=claim_a.sentence_text,
                    claim_b_text=claim_b.sentence_text,
                    contradiction_type="NUMERIC",
                    explanation=(
                        f"The same predicate '{claim_a.predicate_lemma}' is associated with "
                        f"conflicting numeric values: {na['value']} {na['unit']} "
                        f"vs {nb['value']} {nb['unit']} (relative difference: "
                        f"{rel_diff * 100:.1f}%)."
                    ),
                    confidence=1.0,
                    evidence_a=na["raw"],
                    evidence_b=nb["raw"],
                )
    return None


def _check_temporal_contradiction(claim_a, claim_b, norm_a: dict, norm_b: dict) -> Optional[ContradictionAlert]:
    """
    Detect temporal contradictions.

    Fires when two claims share the same subject+predicate but contain
    different normalised (ISO 8601) temporal expressions.
    """
    if not norm_a["temporals"] or not norm_b["temporals"]:
        return None

    if not _claims_are_related(claim_a, claim_b, require_evidence=True):
        return None

    iso_a = [t["iso"] for t in norm_a["temporals"] if t["iso"]]
    iso_b = [t["iso"] for t in norm_b["temporals"] if t["iso"]]

    if not iso_a or not iso_b:
        return None

    # Compare only if they are at the same or compatible resolution
    for ta in iso_a:
        for tb in iso_b:
            if ta != tb:
                # Quick compatibility check: if one is a prefix of the other (e.g.
                # "2024" vs "2024-03") they might not truly conflict
                if ta.startswith(tb) or tb.startswith(ta):
                    continue
                return ContradictionAlert(
                    claim_a_index=claim_a.sentence_index,
                    claim_b_index=claim_b.sentence_index,
                    claim_a_text=claim_a.sentence_text,
                    claim_b_text=claim_b.sentence_text,
                    contradiction_type="TEMPORAL",
                    explanation=(
                        f"Conflicting temporal expressions for the same event "
                        f"(predicate '{claim_a.predicate_lemma}'): "
                        f"'{ta}' vs '{tb}'."
                    ),
                    confidence=1.0,
                    evidence_a=norm_a["temporals"][0]["raw"],
                    evidence_b=norm_b["temporals"][0]["raw"],
                )
    return None


def _check_entity_contradiction(claim_a, claim_b) -> Optional[ContradictionAlert]:
    """
    Detect named-entity contradictions.

    Fires when:
      - Both claims mention entities of the same type (e.g. both PER)
      - They share the same predicate
      - The entity values are different (possible substitution contradiction)

    This is a conservative rule — only fires on the same NE type + same predicate.
    """
    if not claim_a.entities or not claim_b.entities:
        return None
    if not _same_predicate(claim_a, claim_b):
        return None

    for label in claim_a.entities:
        if label not in claim_b.entities:
            continue
        ents_a = set(e.lower() for e in claim_a.entities[label])
        ents_b = set(e.lower() for e in claim_b.entities[label])
        # If both mention entities of the same type but completely different ones
        if ents_a and ents_b and ents_a.isdisjoint(ents_b):
            return ContradictionAlert(
                claim_a_index=claim_a.sentence_index,
                claim_b_index=claim_b.sentence_index,
                claim_a_text=claim_a.sentence_text,
                claim_b_text=claim_b.sentence_text,
                contradiction_type="ENTITY",
                explanation=(
                    f"Conflicting {label} entities associated with the same predicate "
                    f"'{claim_a.predicate_lemma}': {sorted(ents_a)} vs {sorted(ents_b)}."
                ),
                confidence=0.8,
                evidence_a=", ".join(sorted(ents_a)),
                evidence_b=", ".join(sorted(ents_b)),
            )
    return None


# ---------------------------------------------------------------------------
# Layer 1b — WordNet antonymy check (Phase 3 extension)
# ---------------------------------------------------------------------------

def _check_wordnet_contradiction(claim_a, claim_b) -> Optional[ContradictionAlert]:
    """
    Use RoWordNet to detect antonym and negated-synonym contradictions.
    Delegates to WordNetChecker; maps result to ContradictionAlert.
    """
    try:
        from src.pipeline.wordnet_checker import WordNetChecker
        checker = WordNetChecker()
        result = checker.check(claim_a, claim_b)
        if result:
            wn_type = result["type"]  # 'ANTONYM' or 'NEGATED_SYNONYM'
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
# Layer 2 — NLI-based soft detection (Phase 4)
# ---------------------------------------------------------------------------

def _check_nli_contradiction(claim_a, claim_b, threshold: float = 0.65) -> Optional[ContradictionAlert]:
    """
    Use the multilingual NLI module (symanto/xlm-roberta-base-snli-mnli)
    to detect soft linguistic contradictions.

    Only fires when the contradiction probability exceeds `threshold`.
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
                    f"Multilingual NLI model (XLM-RoBERTa) detected a linguistic "
                    f"contradiction (confidence: {score:.2f})."
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
    Detect contradictions between all pairs of claims in a document.

    Parameters
    ----------
    use_wordnet : bool
        If True, run the RoWordNet antonymy/negated-synonym check (Phase 3).
    use_nli : bool
        If True, run the multilingual NLI soft-contradiction check (Phase 4, slower).
    nli_threshold : float
        Minimum NLI contradiction probability to emit an alert.
    """

    def __init__(
        self,
        use_wordnet: bool = True,
        use_nli: bool = False,
        nli_threshold: float = 0.65,
    ):
        self.use_wordnet = use_wordnet
        self.use_nli = use_nli
        self.nli_threshold = nli_threshold

    def detect(self, claims, norm_data: list) -> List[ContradictionAlert]:
        """
        Compare all pairs of claims and return detected contradictions.

        Parameters
        ----------
        claims : List[Claim]
        norm_data : List[dict]
            Normalised attributes for each claim (output of Normalizer).

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
                nli_alert = _check_nli_contradiction(claim_a, claim_b, self.nli_threshold)
                if nli_alert:
                    alerts.append(nli_alert)

        return alerts






