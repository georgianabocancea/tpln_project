"""
evaluator.py
------------
Phase 5 — Evaluation benchmark construction and system scoring.

Provides:
  1. BenchmarkBuilder — creates an evaluation set from MLSUM/XLSum articles
     by injecting known contradictions (numeric, temporal, entity).
  2. Evaluator — runs the full pipeline on the benchmark and reports
     precision, recall, F1 per language and per contradiction type.

Evaluation follows the FEVER methodology for claim-level scoring.
"""

from __future__ import annotations

import json
import logging
import os
import re
import random
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "evaluation")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """A single evaluation sample."""
    sample_id: str
    language: str                         # 'ro' or 'es'
    article_text: str
    injected: bool                        # True = synthetic injection; False = real
    gold_contradictions: List[dict]       # [{type, sentence_a, sentence_b, explanation}]
    source_dataset: str = ""


@dataclass
class EvalResult:
    sample_id: str
    predicted_contradictions: List[dict]
    gold_contradictions: List[dict]
    tp: int = 0
    fp: int = 0
    fn: int = 0


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

# Pairs of (original_substring, replacement, contradiction_type)
# Covers Wikipedia-style Romanian text: years, numbers, months, ordinals.
_NUMERIC_INJECTIONS = [
    # Percentages
    ("10%", "35%", "NUMERIC"),
    ("5%", "22%", "NUMERIC"),
    ("20%", "3%", "NUMERIC"),
    ("50%", "12%", "NUMERIC"),
    # Millions / billions (Romanian)
    ("5 milioane", "20 milioane", "NUMERIC"),
    ("10 milioane", "2 milioane", "NUMERIC"),
    ("100 milioane", "500 milioane", "NUMERIC"),
    ("3 miliarde", "1 miliard", "NUMERIC"),
    ("2 miliarde", "7 miliarde", "NUMERIC"),
    # Raw cardinal numbers (common in Wikipedia infoboxes / dates)
    (" 100 ", " 350 ", "NUMERIC"),
    (" 200 ", " 50 ", "NUMERIC"),
    (" 500 ", " 1200 ", "NUMERIC"),
    (" 1000 ", " 250 ", "NUMERIC"),
    # Years — most common in Wikipedia
    ("2024", "2019", "TEMPORAL"),
    ("2023", "2021", "TEMPORAL"),
    ("2022", "2018", "TEMPORAL"),
    ("2020", "2015", "TEMPORAL"),
    ("2010", "2005", "TEMPORAL"),
    ("2000", "1995", "TEMPORAL"),
    ("1989", "1975", "TEMPORAL"),
    ("1990", "1980", "TEMPORAL"),
    # Centuries
    ("secolul al XIX-lea", "secolul al XVII-lea", "TEMPORAL"),
    ("secolul al XX-lea", "secolul al XVIII-lea", "TEMPORAL"),
    ("secolul al XVIII-lea", "secolul al XVI-lea", "TEMPORAL"),
    # Months (Romanian)
    ("martie", "octombrie", "TEMPORAL"),
    ("ianuarie", "iulie", "TEMPORAL"),
    ("mai", "noiembrie", "TEMPORAL"),
    ("aprilie", "august", "TEMPORAL"),
    ("iunie", "decembrie", "TEMPORAL"),
    ("septembrie", "februarie", "TEMPORAL"),
]


def _inject_contradiction(text: str, seed: int = 42) -> Tuple[str, Optional[dict]]:
    """
    Attempt to inject a single contradiction into an article text.

    Strategy (in priority order):
    1. Replace the SECOND occurrence of a pattern with a conflicting value
       (most reliable — original and modified both visible in the article).
    2. Duplicate a sentence containing the pattern and modify the duplicate.

    Tries all patterns shuffled by seed; returns on first successful injection.
    """
    rng = random.Random(seed)
    shuffled = _NUMERIC_INJECTIONS[:]
    rng.shuffle(shuffled)

    # Priority 1: patterns with two or more occurrences in the text
    for original, replacement, ctype in shuffled:
        lower_text = text.lower()
        orig_lower = original.lower()
        if orig_lower not in lower_text:
            continue

        idx = lower_text.find(orig_lower)
        second_idx = lower_text.find(orig_lower, idx + len(orig_lower))
        if second_idx == -1:
            continue

        # Preserve original casing at the replacement site
        before = text[:second_idx]
        after  = text[second_idx:]
        after_modified = after.replace(text[second_idx:second_idx + len(original)], replacement, 1)
        return before + after_modified, {
            "type": ctype,
            "original_value": original,
            "injected_value": replacement,
            "method": "second_occurrence_replacement",
        }

    # Priority 2: duplicate a sentence then modify the copy
    for original, replacement, ctype in shuffled:
        orig_lower = original.lower()
        if orig_lower not in text.lower():
            continue

        # Find the sentence containing the pattern
        for sent in text.split("."):
            if orig_lower in sent.lower():
                modified_sent = sent + "."
                # Case-insensitive replace inside the duplicated sentence
                modified = re.sub(
                    re.escape(original), replacement, modified_sent, count=1, flags=re.IGNORECASE
                )
                if modified != modified_sent:
                    injected_text = text.rstrip() + " " + modified
                    return injected_text, {
                        "type": ctype,
                        "original_value": original,
                        "injected_value": replacement,
                        "method": "sentence_duplication_with_modification",
                    }

    return text, None


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------

class BenchmarkBuilder:
    """
    Build an evaluation benchmark from open Romanian datasets.

    Draws articles from MLSUM and XLSum Romanian, injects known contradictions
    into a subset, and saves matching gold labels.

    Parameters
    ----------
    n_injected : int
        Number of synthetic (injected) samples to create.
    n_real : int
        Number of real (non-injected) samples to include (labelled as
        no-contradiction for precision measurement).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_injected: int = 100, n_real: int = 50, seed: int = 42):
        self.n_injected = n_injected
        self.n_real = n_real
        self.seed = seed
        self.samples: List[BenchmarkSample] = []

    def build(self) -> List[BenchmarkSample]:
        """Load datasets, inject contradictions, assemble benchmark."""
        from datasets import load_dataset

        logger.info("Building evaluation benchmark…")
        cache_wiki = os.path.join(PROJECT_ROOT, "data", "raw", "wiki_ro")

        # Romanian Wikipedia — primary corpus for both injected and real samples
        try:
            wiki = load_dataset(
                "wikimedia/wikipedia", "20231101.ro",
                split="train",
                cache_dir=cache_wiki,
            )
        except Exception as exc:
            logger.warning("Could not load Romanian Wikipedia: %s — using empty list.", exc)
            wiki = []

        rng = random.Random(self.seed)
        articles = list(wiki)
        rng.shuffle(articles)

        # --- Injected samples ---
        injected_count = 0
        for row in articles:
            if injected_count >= self.n_injected:
                break
            text = row.get("text", "").strip()
            if len(text) < 200:
                continue
            injected_text, injection_meta = _inject_contradiction(
                text, seed=self.seed + injected_count
            )
            if injection_meta is None:
                continue

            self.samples.append(BenchmarkSample(
                sample_id=f"injected_{injected_count:04d}",
                language="ro",
                article_text=injected_text,
                injected=True,
                gold_contradictions=[{
                    "type": injection_meta["type"],
                    "method": injection_meta["method"],
                    "original_value": injection_meta["original_value"],
                    "injected_value": injection_meta["injected_value"],
                }],
                source_dataset="wiki_ro",
            ))
            injected_count += 1

        logger.info("Created %d injected samples.", injected_count)

        # --- Real (no-contradiction) samples — use remaining Wikipedia articles ---
        real_count = 0
        for row in articles[injected_count:]:
            if real_count >= self.n_real:
                break
            text = row.get("text", "").strip()
            if len(text) < 150:
                continue

            self.samples.append(BenchmarkSample(
                sample_id=f"real_{real_count:04d}",
                language="ro",
                article_text=text[:2000],   # cap length for speed
                injected=False,
                gold_contradictions=[],
                source_dataset="wiki_ro",
            ))
            real_count += 1

        logger.info("Added %d real (no-contradiction) samples.", real_count)
        return self.samples

    def save(self, path: Optional[str] = None) -> str:
        """Save the benchmark to a JSON file."""
        if path is None:
            os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
            path = os.path.join(EVAL_OUTPUT_DIR, "benchmark.json")

        data = [asdict(s) for s in self.samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Benchmark saved to %s (%d samples).", path, len(data))
        return path

    @staticmethod
    def load(path: str) -> List[BenchmarkSample]:
        """Load a previously saved benchmark."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [BenchmarkSample(**d) for d in data]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Run the pipeline on all benchmark samples and compute evaluation metrics.

    Scoring strategy (per-sample):
      - Gold has ≥1 contradiction AND pipeline finds ≥1 → True Positive
      - Gold has ≥1 contradiction AND pipeline finds 0  → False Negative
      - Gold has 0 contradictions AND pipeline finds ≥1 → False Positive
      - Gold has 0 contradictions AND pipeline finds 0  → True Negative

    Reports precision, recall, F1 overall and broken down by
    contradiction type and by injected vs real.
    """

    def __init__(self, pipeline=None, backend: str = "spacy", use_nli: bool = False):
        if pipeline is None:
            from src.pipeline.pipeline import RomanianPipeline
            self.pipeline = RomanianPipeline(backend=backend, use_nli=use_nli)
        else:
            self.pipeline = pipeline

    def run(self, samples: List[BenchmarkSample]) -> dict:
        """
        Evaluate the pipeline on all samples.

        Returns
        -------
        dict with keys: overall, by_type, injected, real, per_sample
        """
        results: List[EvalResult] = []

        for i, sample in enumerate(samples):
            logger.info(
                "Evaluating sample %d/%d (%s)…",
                i + 1, len(samples), sample.sample_id
            )
            try:
                pipeline_result = self.pipeline.run(sample.article_text)
                predicted = [c.to_dict() for c in pipeline_result.contradictions]
            except Exception as exc:
                logger.error("Pipeline failed on sample %s: %s", sample.sample_id, exc)
                predicted = []

            gold_has_contradiction = len(sample.gold_contradictions) > 0
            pred_has_contradiction = len(predicted) > 0

            tp = 1 if gold_has_contradiction and pred_has_contradiction else 0
            fp = 1 if not gold_has_contradiction and pred_has_contradiction else 0
            fn = 1 if gold_has_contradiction and not pred_has_contradiction else 0

            results.append(EvalResult(
                sample_id=sample.sample_id,
                predicted_contradictions=predicted,
                gold_contradictions=sample.gold_contradictions,
                tp=tp, fp=fp, fn=fn,
            ))

        return self._aggregate(results, samples)

    @staticmethod
    def _aggregate(results: List[EvalResult], samples: List[BenchmarkSample]) -> dict:
        """Compute precision, recall, F1 overall and by subset."""

        def prf(tp, fp, fn):
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            return round(prec, 4), round(rec, 4), round(f1, 4)

        total_tp = sum(r.tp for r in results)
        total_fp = sum(r.fp for r in results)
        total_fn = sum(r.fn for r in results)

        prec, rec, f1 = prf(total_tp, total_fp, total_fn)

        # Split by injected vs real
        sample_map = {s.sample_id: s for s in samples}

        def subset_prf(filter_fn):
            stp = sum(r.tp for r in results if filter_fn(sample_map[r.sample_id]))
            sfp = sum(r.fp for r in results if filter_fn(sample_map[r.sample_id]))
            sfn = sum(r.fn for r in results if filter_fn(sample_map[r.sample_id]))
            return prf(stp, sfp, sfn)

        inj_p, inj_r, inj_f = subset_prf(lambda s: s.injected)
        real_p, real_r, real_f = subset_prf(lambda s: not s.injected)

        # Type-level breakdown (based on gold labels of TP samples)
        type_stats: Dict[str, Dict[str, int]] = {}
        for r, s in zip(results, samples):
            for gc in s.gold_contradictions:
                t = gc.get("type", "UNKNOWN")
                if t not in type_stats:
                    type_stats[t] = {"tp": 0, "fp": 0, "fn": 0}
                type_stats[t]["tp"] += r.tp
                type_stats[t]["fn"] += r.fn

        by_type = {}
        for t, counts in type_stats.items():
            p, rc, f = prf(counts["tp"], counts.get("fp", 0), counts["fn"])
            by_type[t] = {"precision": p, "recall": rc, "f1": f, **counts}

        report = {
            "overall": {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "total_samples": len(results),
            },
            "injected": {"precision": inj_p, "recall": inj_r, "f1": inj_f},
            "real": {"precision": real_p, "recall": real_r, "f1": real_f},
            "by_type": by_type,
            "per_sample": [
                {
                    "sample_id": r.sample_id,
                    "tp": r.tp, "fp": r.fp, "fn": r.fn,
                    "predicted": len(r.predicted_contradictions),
                    "gold": len(r.gold_contradictions),
                }
                for r in results
            ],
        }
        return report

    def save_report(self, report: dict, path: Optional[str] = None) -> str:
        """Save evaluation report to JSON."""
        if path is None:
            os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
            path = os.path.join(EVAL_OUTPUT_DIR, "eval_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Evaluation report saved to %s.", path)
        return path

    @staticmethod
    def print_report(report: dict):
        """Pretty-print the evaluation report to stdout."""
        print("\n" + "=" * 60)
        print("  Evaluation Report — Romanian Contradiction Detection")
        print("=" * 60)
        ov = report["overall"]
        print(f"\n  Overall  —  P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}")
        print(f"  Samples: {ov['total_samples']} | TP={ov['tp']} FP={ov['fp']} FN={ov['fn']}")

        inj = report["injected"]
        print(f"\n  Injected —  P={inj['precision']:.3f}  R={inj['recall']:.3f}  F1={inj['f1']:.3f}")
        real = report["real"]
        print(f"  Real     —  P={real['precision']:.3f}  R={real['recall']:.3f}  F1={real['f1']:.3f}")

        print("\n  By contradiction type:")
        for t, stats in report["by_type"].items():
            print(
                f"    {t:12s}  P={stats['precision']:.3f}  "
                f"R={stats['recall']:.3f}  F1={stats['f1']:.3f}"
            )
        print("=" * 60 + "\n")

