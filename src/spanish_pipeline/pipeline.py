"""
pipeline.py
-----------
Top-level orchestrator for the Spanish contradiction detection pipeline.

  SpanishPreprocessor → ClaimExtractor → Normalizer → ContradictionDetector

Usage
-----
    from src.spanish_pipeline.pipeline import SpanishPipeline

    pipeline = SpanishPipeline()
    result = pipeline.run(article_text)
    print(result.to_json())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List

from src.spanish_pipeline.preprocessor import SpanishPreprocessor
from src.spanish_pipeline.claim_extractor import ClaimExtractor
from src.spanish_pipeline.normalizer import Normalizer
from src.spanish_pipeline.contradiction_detector import (
    ContradictionDetector,
    ContradictionAlert,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    article_text: str
    num_sentences: int
    num_claims: int
    contradictions: List[ContradictionAlert]

    def to_dict(self) -> dict:
        return {
            "num_sentences":     self.num_sentences,
            "num_claims":        self.num_claims,
            "num_contradictions": len(self.contradictions),
            "contradictions":    [c.to_dict() for c in self.contradictions],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def __repr__(self):
        return (
            f"PipelineResult("
            f"sentences={self.num_sentences}, "
            f"claims={self.num_claims}, "
            f"contradictions={len(self.contradictions)})"
        )


class SpanishPipeline:
    """
    End-to-end Spanish contradiction detection pipeline.

    Parameters
    ----------
    backend : str
        Preprocessing backend: ``'auto'`` (default), ``'spacy'``, or
        ``'stanza'``.
    use_wordnet : bool
        Enable NLTK Spanish WordNet antonymy / negated-synonym detection.
    use_nli : bool
        Enable NLI-based soft contradiction detection (slower, multilingual
        XLM-RoBERTa model — the same one used by the Romanian pipeline).
    nli_threshold : float
        NLI contradiction confidence threshold (0–1).

    Examples
    --------
    >>> pipeline = SpanishPipeline()
    >>> result = pipeline.run(
    ...     "El gobierno anunció que la economía creció un 3 % en 2023. "
    ...     "Según otro informe, la economía se contrajo un 5 % en 2023."
    ... )
    >>> print(result.to_json())
    """

    def __init__(
        self,
        backend: str = "auto",
        use_wordnet: bool = True,
        use_nli: bool = False,
        nli_threshold: float = 0.65,
    ):
        self.preprocessor = SpanishPreprocessor(backend=backend)
        self.extractor    = ClaimExtractor()
        self.normalizer   = Normalizer()
        self.detector     = ContradictionDetector(
            use_wordnet=use_wordnet,
            use_nli=use_nli,
            nli_threshold=nli_threshold,
        )

    def run(self, article_text: str) -> PipelineResult:
        """
        Process a single Spanish news article end-to-end.

        Parameters
        ----------
        article_text : str
            The full text of the article.

        Returns
        -------
        PipelineResult
        """
        logger.info("Preprocessing article (%d chars)…", len(article_text))
        sentences = self.preprocessor.process(article_text)

        logger.info("Extracting claims from %d sentences…", len(sentences))
        claims = self.extractor.extract(sentences)

        logger.info("Normalising %d claims…", len(claims))
        norm_data = self.normalizer.normalize_claims(claims)

        logger.info("Running contradiction detection…")
        contradictions = self.detector.detect(claims, norm_data)

        logger.info(
            "Done — %d claim(s), %d contradiction(s) found.",
            len(claims),
            len(contradictions),
        )

        return PipelineResult(
            article_text=article_text,
            num_sentences=len(sentences),
            num_claims=len(claims),
            contradictions=contradictions,
        )


