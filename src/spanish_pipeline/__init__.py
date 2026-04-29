"""
Spanish NLP pipeline for news contradiction detection.

Sub-modules
-----------
preprocessor        : tokenisation, POS, lemma, NER, dep-parse (spaCy / Stanza)
claim_extractor     : SPO triplet extraction with numeric / temporal / NE attributes
normalizer          : canonical numeric and temporal normalisation
wordnet_checker     : lexical-semantic antonymy check via NLTK WordNet (Spanish)
contradiction_detector : rule-based + NLI contradiction detection
pipeline            : end-to-end SpanishPipeline orchestrator
"""

__all__ = ["SpanishPipeline", "PipelineResult"]


def __getattr__(name):
    if name in ("SpanishPipeline", "PipelineResult"):
        from src.spanish_pipeline.pipeline import SpanishPipeline, PipelineResult  # noqa: F401
        return {"SpanishPipeline": SpanishPipeline, "PipelineResult": PipelineResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


