"""
pipeline.py
-----------
Top-level orchestrator that ties together all pipeline components:

  RomanianPreprocessor → ClaimExtractor → Normalizer → ContradictionDetector

Usage
-----
    from src.pipeline.pipeline import RomanianPipeline

    pipeline = RomanianPipeline()
    result = pipeline.run(article_text)
    print(result)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import List

from src.pipeline.preprocessor import RomanianPreprocessor
from src.pipeline.claim_extractor import ClaimExtractor
from src.pipeline.normalizer import Normalizer
from src.pipeline.contradiction_detector import ContradictionDetector, ContradictionAlert

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    article_text: str
    num_sentences: int
    num_claims: int
    contradictions: List[ContradictionAlert]

    def to_dict(self) -> dict:
        return {
            "num_sentences": self.num_sentences,
            "num_claims": self.num_claims,
            "num_contradictions": len(self.contradictions),
            "contradictions": [c.to_dict() for c in self.contradictions],
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


class RomanianPipeline:
    """
    End-to-end Romanian contradiction detection pipeline.

    Parameters
    ----------
    backend : str
        Preprocessing backend: 'auto', 'teprolin', 'spacy', or 'stanza'.
    use_wordnet : bool
        Enable RoWordNet antonymy / negated-synonym detection (Phase 3).
    use_nli : bool
        Enable NLI-based soft contradiction detection (Phase 4, slower).
    nli_threshold : float
        NLI contradiction confidence threshold (0–1).
    """

    def __init__(
        self,
        backend: str = "auto",
        use_wordnet: bool = True,
        use_nli: bool = False,
        nli_threshold: float = 0.65,
    ):
        self.preprocessor = RomanianPreprocessor(backend=backend)
        self.extractor = ClaimExtractor()
        self.normalizer = Normalizer()
        self.detector = ContradictionDetector(
            use_wordnet=use_wordnet,
            use_nli=use_nli,
            nli_threshold=nli_threshold,
        )

    def run(self, article_text: str) -> PipelineResult:
        """
        Process a single Romanian news article end-to-end.

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

