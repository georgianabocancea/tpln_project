"""
run_evaluation.py
-----------------
CLI script for Phase 5 evaluation of the Romanian and Spanish pipelines.

Usage:
    # Romanian (default)
    python scripts/run_evaluation.py [--n-injected 100] [--n-real 50]
                                     [--backend spacy] [--use-nli]
                                     [--output data/processed/evaluation/]

    # Spanish
    python scripts/run_evaluation.py --language spanish
                                     [--n-injected 100] [--n-real 50]
                                     [--backend spacy] [--use-nli]

    # Both languages
    python scripts/run_evaluation.py --language both

Outputs:
    - data/processed/evaluation/benchmark.json      (benchmark samples)
    - data/processed/evaluation/eval_report.json    (Romanian metrics)
    - data/processed/evaluation/eval_report_es.json (Spanish metrics)
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_divider(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_romanian_evaluation(args):
    """Build benchmark and evaluate the Romanian pipeline."""
    _print_divider("Romanian Pipeline — Evaluation")

    from src.pipeline.evaluator import BenchmarkBuilder, Evaluator
    from src.pipeline.pipeline import RomanianPipeline

    # --- Build or load benchmark ---
    if args.load_benchmark and os.path.isfile(args.load_benchmark):
        logger.info("Loading existing benchmark from %s", args.load_benchmark)
        samples = BenchmarkBuilder.load(args.load_benchmark)
    else:
        logger.info(
            "Building Romanian benchmark (injected=%d, real=%d)…",
            args.n_injected, args.n_real,
        )
        builder = BenchmarkBuilder(
            n_injected=args.n_injected,
            n_real=args.n_real,
            seed=args.seed,
        )
        samples = builder.build()
        benchmark_path = builder.save()
        logger.info("Benchmark saved to %s", benchmark_path)

    if not samples:
        logger.error("No samples available — check that MLSUM/XLSum are accessible.")
        return None

    pipeline = RomanianPipeline(
        backend=args.backend,
        use_wordnet=args.use_wordnet,
        use_nli=args.use_nli,
        nli_threshold=args.nli_threshold,
    )

    evaluator = Evaluator(pipeline=pipeline)
    report = evaluator.run(samples)
    report_path = evaluator.save_report(report)
    Evaluator.print_report(report)
    logger.info("Romanian report saved to %s", report_path)
    return report


def run_spanish_evaluation(args):
    """Build a simple benchmark and evaluate the Spanish pipeline."""
    _print_divider("Spanish Pipeline — Evaluation")

    from src.spanish_pipeline.pipeline import SpanishPipeline

    pipeline = SpanishPipeline(
        backend=args.es_backend,
        use_wordnet=args.use_wordnet,
        use_nli=args.use_nli,
        nli_threshold=args.nli_threshold,
    )

    # Built-in Spanish benchmark: injected contradiction pairs + clean articles
    injected_articles = [
        # Numeric contradictions
        (
            "La empresa reportó ganancias de 5 millones de euros en el primer trimestre de 2024. "
            "El director general afirmó que las ganancias ascendieron a 12 millones de euros en el primer trimestre de 2024.",
            True,
        ),
        (
            "El presupuesto estatal fue de 50.000 millones en 2023. "
            "El ministro de economía confirmó que el presupuesto fue de 75.000 millones en 2023.",
            True,
        ),
        # Temporal contradictions
        (
            "El gobierno aprobó la ley el 15 de marzo de 2023. "
            "Según fuentes oficiales, la ley fue aprobada en septiembre de 2024.",
            True,
        ),
        (
            "La cumbre climática tuvo lugar en noviembre de 2022. "
            "El comunicado señala que la cumbre se celebró en enero de 2023.",
            True,
        ),
        # Antonym contradictions
        (
            "La economía creció significativamente en el último trimestre, según datos del INE. "
            "Los analistas sostienen que la economía se contrajo en el mismo período.",
            True,
        ),
        (
            "El presidente aprobó la reforma educativa ante el Congreso. "
            "Según informes internos, el presidente rechazó la reforma educativa en la misma sesión.",
            True,
        ),
        # Clean articles (no contradiction)
        (
            "España registró un crecimiento económico del 3% en 2023. "
            "El Banco de España mantuvo el tipo de interés de referencia en el 4%.",
            False,
        ),
        (
            "El equipo nacional ganó el partido por 2 a 1 ante Francia. "
            "El seleccionador destacó la actuación del portero durante la rueda de prensa.",
            False,
        ),
        (
            "La tasa de desempleo bajó al 11% en el segundo trimestre. "
            "El gobierno atribuyó la mejora a las nuevas políticas de formación.",
            False,
        ),
        (
            "El festival de cine abrió sus puertas el viernes con más de 200 películas. "
            "Más de 5.000 visitantes asistieron al estreno de la película inaugural.",
            False,
        ),
    ]

    # Add more samples up to n_injected / n_real if flags were set
    import random
    rng = random.Random(args.seed)

    true_samples  = [(t, l) for t, l in injected_articles if     l]
    false_samples = [(t, l) for t, l in injected_articles if not l]

    # Repeat to reach the requested counts
    while len(true_samples)  < args.n_injected:
        true_samples.extend(true_samples)
    while len(false_samples) < args.n_real:
        false_samples.extend(false_samples)

    benchmark = (
        rng.sample(true_samples,  min(args.n_injected, len(true_samples)))
        + rng.sample(false_samples, min(args.n_real,     len(false_samples)))
    )
    rng.shuffle(benchmark)

    # Evaluate
    tp = fp = fn = tn = 0
    results = []
    for article_text, has_contradiction in benchmark:
        result = pipeline.run(article_text)
        predicted = len(result.contradictions) > 0
        expected  = has_contradiction

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

        results.append({
            "article": article_text[:80] + "…",
            "expected": expected,
            "predicted": predicted,
            "contradictions_found": len(result.contradictions),
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    report = {
        "language": "spanish",
        "pipeline_config": {
            "backend": args.es_backend,
            "use_wordnet": args.use_wordnet,
            "use_nli": args.use_nli,
            "nli_threshold": args.nli_threshold,
        },
        "overall": {
            "total_samples": len(benchmark),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        },
        "samples": results,
    }

    # Print summary
    print(f"\n  Spanish pipeline evaluation results:")
    print(f"    Total samples : {len(benchmark)}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Precision : {precision:.3f}")
    print(f"    Recall    : {recall:.3f}")
    print(f"    F1        : {f1:.3f}")

    # Save report
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report_es.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("Spanish report saved to %s", report_path)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Romanian and/or Spanish contradiction detection pipeline."
    )
    parser.add_argument(
        "--language",
        choices=["romanian", "spanish", "both"],
        default="romanian",
        help="Which pipeline to evaluate (default: romanian).",
    )

    # Common options
    parser.add_argument("--n-injected", type=int, default=100,
                        help="Number of injected contradiction samples")
    parser.add_argument("--n-real", type=int, default=50,
                        help="Number of real (no-contradiction) samples")
    parser.add_argument("--use-wordnet", action="store_true", default=True,
                        help="Enable WordNet checks")
    parser.add_argument("--use-nli", action="store_true",
                        help="Enable NLI soft detection (slow)")
    parser.add_argument("--nli-threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)

    # Romanian-specific
    parser.add_argument("--backend",
                        choices=["auto", "spacy", "stanza", "teprolin"],
                        default="spacy",
                        help="Romanian preprocessing backend")
    parser.add_argument("--load-benchmark", type=str, default=None,
                        help="Path to existing benchmark.json (skip building, Romanian only)")

    # Spanish-specific
    parser.add_argument("--es-backend",
                        choices=["auto", "spacy", "stanza"],
                        default="spacy",
                        help="Spanish preprocessing backend")

    args = parser.parse_args()

    if args.language in ("romanian", "both"):
        run_romanian_evaluation(args)

    if args.language in ("spanish", "both"):
        run_spanish_evaluation(args)

    _print_divider("Evaluation complete.")


if __name__ == "__main__":
    main()

