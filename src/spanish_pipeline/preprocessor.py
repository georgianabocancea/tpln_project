"""
preprocessor.py
---------------
Phase 1 — Preprocessing and linguistic analysis for Spanish text.

Two backends in priority order:
  1. spaCy  es_core_news_lg  (tokenisation, POS, lemma, NER, dep-parse)
  2. Stanza es               (fallback / cross-validation)

Output: List[Sentence], each containing List[Token].
The Token / Sentence dataclasses are *identical* to the Romanian ones so the
rest of the pipeline (ClaimExtractor, Normalizer, ContradictionDetector) is
fully re-usable without modification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STANZA_ES_DIR = os.path.join(PROJECT_ROOT, "models", "stanza_es")


# ---------------------------------------------------------------------------
# Data structures  (mirror of Romanian preprocessor — shared interface)
# ---------------------------------------------------------------------------


@dataclass
class Token:
    index: int          # 1-based within sentence
    text: str
    lemma: str
    pos: str            # Universal POS tag
    dep: str            # UD dependency relation label
    head: int           # 1-based index of head token (0 = root)
    ner: str = "O"      # BIO NER tag


@dataclass
class Sentence:
    index: int          # 0-based within document
    text: str
    tokens: List[Token] = field(default_factory=list)


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("es_core_news_lg")
        except OSError:
            logger.warning(
                "es_core_news_lg not found — trying es_core_news_md / es_core_news_sm"
            )
            for model in ("es_core_news_md", "es_core_news_sm"):
                try:
                    _spacy_nlp = spacy.load(model)
                    break
                except OSError:
                    continue
        if _spacy_nlp is None:
            raise RuntimeError(
                "No Spanish spaCy model found. "
                "Install one with: python -m spacy download es_core_news_lg"
            )
    return _spacy_nlp


def _spacy_process(text: str) -> List[Sentence]:
    nlp = _get_spacy()
    doc = nlp(text)
    sentences: List[Sentence] = []

    # Build token-level NER lookup from doc.ents
    ner_map: dict[int, str] = {}
    for ent in doc.ents:
        for i, tok in enumerate(ent):
            prefix = "B-" if i == 0 else "I-"
            ner_map[tok.i] = f"{prefix}{ent.label_}"

    for sent_idx, sent in enumerate(doc.sents):
        tokens: List[Token] = []
        for local_idx, tok in enumerate(sent):
            head_idx = 0 if tok.head == tok else (tok.head.i - sent.start + 1)
            tokens.append(Token(
                index=local_idx + 1,
                text=tok.text,
                lemma=tok.lemma_,
                pos=tok.pos_,
                dep=tok.dep_,
                head=head_idx,
                ner=ner_map.get(tok.i, "O"),
            ))
        sentences.append(Sentence(index=sent_idx, text=sent.text, tokens=tokens))

    return sentences


# ---------------------------------------------------------------------------
# Stanza backend
# ---------------------------------------------------------------------------

_stanza_nlp = None


def _get_stanza():
    global _stanza_nlp
    if _stanza_nlp is None:
        import stanza
        stanza_kwargs: dict = {"lang": "es", "verbose": False, "use_gpu": False}
        if os.path.isdir(STANZA_ES_DIR):
            stanza_kwargs["dir"] = STANZA_ES_DIR
        _stanza_nlp = stanza.Pipeline(**stanza_kwargs)
    return _stanza_nlp


def _stanza_process(text: str) -> List[Sentence]:
    nlp = _get_stanza()
    doc = nlp(text)
    sentences: List[Sentence] = []

    for sent_idx, sent in enumerate(doc.sentences):
        tokens: List[Token] = []

        # Build NER lookup: word id (1-based) → BIO tag
        ner_map: dict[int, str] = {}
        for ent in sent.ents:
            for i, tok in enumerate(ent.tokens):
                prefix = "B-" if i == 0 else "I-"
                ner_map[tok.id[0]] = f"{prefix}{ent.type}"

        for word in sent.words:
            tokens.append(Token(
                index=word.id,
                text=word.text,
                lemma=word.lemma or word.text,
                pos=word.upos or "X",
                dep=word.deprel or "",
                head=word.head,
                ner=ner_map.get(word.id, "O"),
            ))

        sentences.append(Sentence(index=sent_idx, text=sent.text, tokens=tokens))

    return sentences


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SpanishPreprocessor:
    """
    Main entry point for Spanish text preprocessing.

    Parameters
    ----------
    backend : str
        One of ``'spacy'``, ``'stanza'``, or ``'auto'``.
        ``'auto'`` (default) tries spaCy first, falls back to Stanza.
    """

    def __init__(self, backend: str = "auto"):
        assert backend in {"auto", "spacy", "stanza"}, (
            f"Unknown backend '{backend}'. Choose from: auto, spacy, stanza"
        )
        self.backend = backend

    def process(self, text: str) -> List[Sentence]:
        """
        Preprocess a Spanish text string and return annotated sentences.

        Parameters
        ----------
        text : str
            Raw Spanish article text.

        Returns
        -------
        List[Sentence]
        """
        text = text.strip()
        if not text:
            return []

        if self.backend == "spacy":
            return _spacy_process(text)
        elif self.backend == "stanza":
            return _stanza_process(text)
        else:  # auto
            try:
                result = _spacy_process(text)
                if result:
                    return result
            except Exception as exc:
                logger.warning("spaCy failed, falling back to Stanza: %s", exc)
            return _stanza_process(text)

    def process_with_all(self, text: str) -> dict:
        """
        Run both backends and return results for cross-validation / debugging.

        Returns
        -------
        dict with keys ``'spacy'`` and ``'stanza'``
        """
        results: dict = {}
        for name, fn in [("spacy", _spacy_process), ("stanza", _stanza_process)]:
            try:
                results[name] = fn(text)
            except Exception as exc:
                logger.error("%s failed: %s", name, exc)
                results[name] = []
        return results


