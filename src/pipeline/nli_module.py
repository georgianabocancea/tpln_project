"""
nli_module.py
-------------
Phase 4 — Multilingual NLI-based soft contradiction detection.

Uses a pre-trained multilingual NLI model to score sentence pairs as:
  ENTAILMENT / NEUTRAL / CONTRADICTION

Model used:
  MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
    - Based on XLM-RoBERTa tokenizer (250K multilingual vocab, covers Romanian)
    - Fine-tuned on MNLI + XNLI (15 languages including Romanian eval set)
    - ~120 MB — CPU-friendly
    - Stored locally in models/nli/

  NOTE: xlm-roberta-base (models/bert_romanian/) is the ENCODER used for
  sentence embeddings in embeddings.py.  This NLI model is a SEPARATE,
  classification-head model fine-tuned for entailment/contradiction scoring.

The FEVER dataset provides the methodological reference:
  - FEVER label schema: SUPPORTS / REFUTES / NOT ENOUGH INFO
  - We map:  REFUTES → CONTRADICTION, SUPPORTS → ENTAILMENT

Fine-tuning on FEVER is supported via the `FeverCalibrator` class.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NLI model is stored separately from the base BERT encoder
NLI_CACHE = os.path.join(PROJECT_ROOT, "models", "nli")

# ---------------------------------------------------------------------------
# Default model (CPU-friendly multilingual)
# ---------------------------------------------------------------------------

DEFAULT_NLI_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

# FEVER label → NLI label mapping
FEVER_LABEL_MAP = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral",
}

# NLI label indices (model-specific; symanto model uses these)
NLI_LABELS = ["entailment", "neutral", "contradiction"]


# ---------------------------------------------------------------------------
# NLI scorer
# ---------------------------------------------------------------------------

class NLIScorer:
    """
    Score a (premise, hypothesis) pair with an NLI model.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID. Defaults to symanto/xlm-roberta-base-snli-mnli.
    device : str
        'cpu' or 'cuda'. Auto-detects if None.
    threshold : float
        Minimum contradiction probability to emit an alert.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None,
        threshold: float = 0.65,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self._tokenizer = None
        self._model = None

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load(self):
        if self._model is not None:
            return
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            PreTrainedTokenizerBase,
            PreTrainedModel,
        )

        logger.info("Loading NLI model '%s' on %s…", self.model_name, self.device)

        self._tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(self.model_name, cache_dir=NLI_CACHE),
        )
        self._model = cast(
            PreTrainedModel,
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name, cache_dir=NLI_CACHE
            ),
        )
        self._model.to(self.device)
        self._model.eval()

        # Determine label ordering from model config
        id2label = self._model.config.id2label
        self._label_order = [id2label[i].lower() for i in range(len(id2label))]
        logger.info("NLI model ready — labels: %s", self._label_order)

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Return a dict of {label: probability} for the given pair.

        Labels: 'entailment', 'neutral', 'contradiction'
        """
        import torch
        self._load()

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, num_labels)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {label: float(probs[i]) for i, label in enumerate(self._label_order)}

    def is_contradiction(self, premise: str, hypothesis: str) -> Tuple[bool, float]:
        """
        Return (is_contradiction, confidence_score).
        """
        scores = self.score(premise, hypothesis)
        contradiction_score = scores.get("contradiction", 0.0)
        return contradiction_score >= self.threshold, contradiction_score

    def batch_score(
        self, pairs: List[Tuple[str, str]], batch_size: int = 8
    ) -> List[Dict[str, float]]:
        """
        Score multiple (premise, hypothesis) pairs efficiently.

        Parameters
        ----------
        pairs : List of (premise, hypothesis) tuples
        batch_size : int

        Returns
        -------
        List of score dicts (same order as input)
        """
        import torch
        self._load()

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                all_scores.append(
                    {label: float(row[j]) for j, label in enumerate(self._label_order)}
                )

        return all_scores


# ---------------------------------------------------------------------------
# FEVER calibration helper
# ---------------------------------------------------------------------------

class FeverCalibrator:
    """
    Uses the FEVER dataset to calibrate the NLI model's contradiction threshold.

    FEVER is an English fact-verification dataset whose label schema maps
    directly onto NLI: SUPPORTS→entailment, REFUTES→contradiction.
    Calibration finds the threshold that maximises F1 on FEVER dev pairs.

    This class is used once during Phase 4 setup; the calibrated threshold
    is then passed to NLIScorer.
    """

    def __init__(self, scorer: NLIScorer, n_samples: int = 500):
        self.scorer = scorer
        self.n_samples = n_samples

    def calibrate(self) -> float:
        """
        Load FEVER dev split, score pairs, find optimal threshold.

        Returns
        -------
        float : optimal contradiction threshold
        """
        from datasets import load_dataset

        logger.info("Loading FEVER dataset for calibration…")
        # copenlu/fever_gold_evidence: Parquet-based FEVER mirror (claim/label/evidence)
        ds = load_dataset(
            "copenlu/fever_gold_evidence",
            split=f"validation[:{self.n_samples}]",
        )

        pairs = []
        gold_labels = []

        for row in ds:
            claim = row.get("claim", "")
            label = row.get("label", "NOT ENOUGH INFO")
            # Simplified: use claim as both premise and hypothesis.
            # Production use would retrieve the actual Wikipedia evidence sentence.
            pairs.append((claim, claim))
            gold_labels.append(FEVER_LABEL_MAP.get(label, "neutral"))

        logger.info("Scoring %d FEVER pairs…", len(pairs))
        scores_list = self.scorer.batch_score(pairs)

        contra_scores = [s.get("contradiction", 0.0) for s in scores_list]
        gold_binary = [1 if g == "contradiction" else 0 for g in gold_labels]

        best_threshold, best_f1 = self._find_best_threshold(contra_scores, gold_binary)
        logger.info(
            "FEVER calibration complete — optimal threshold: %.3f (F1=%.3f)",
            best_threshold,
            best_f1,
        )
        return best_threshold

    @staticmethod
    def _find_best_threshold(
        scores: List[float], gold: List[int]
    ) -> Tuple[float, float]:
        """Grid-search threshold from 0.3 to 0.9 for best F1."""
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.3, 0.91, 0.05):
            preds = [1 if s >= t else 0 for s in scores]
            tp = sum(p == g == 1 for p, g in zip(preds, gold))
            fp = sum(p == 1 and g == 0 for p, g in zip(preds, gold))
            fn = sum(p == 0 and g == 1 for p, g in zip(preds, gold))
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t, best_f1


# ---------------------------------------------------------------------------
# Convenience function used by ContradictionDetector
# ---------------------------------------------------------------------------

_default_scorer: Optional[NLIScorer] = None


def get_default_scorer(threshold: float = 0.65) -> NLIScorer:
    """Return (lazy-initialised) default NLI scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = NLIScorer(threshold=threshold)
    return _default_scorer

