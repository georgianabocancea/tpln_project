"""
preprocessor.py
---------------
Phase 2 — Preprocessing and linguistic analysis for Romanian text.

Uses three backends in priority order:
  1. Teprolin REST API  (tokenisation, POS, lemma, NER, dep-parse)
  2. spaCy ro_core_news_lg  (fallback / cross-validation)
  3. Stanza Romanian  (fallback / cross-validation)

The output of every method is a list of Sentence dataclasses, each
containing a list of Token dataclasses.
"""

from __future__ import annotations

import os
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STANZA_DIR = os.path.join(PROJECT_ROOT, "models", "stanza")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Token:
    index: int          # 1-based within sentence
    text: str
    lemma: str
    pos: str            # Universal POS tag
    dep: str            # dependency relation label
    head: int           # 1-based index of head token (0 = root)
    ner: str = "O"      # BIO NER tag


@dataclass
class Sentence:
    index: int          # 0-based within document
    text: str
    tokens: List[Token] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Teprolin backend
# ---------------------------------------------------------------------------

TEPROLIN_URL = "http://localhost:5000/process"
TEPROLIN_TIMEOUT = 20  # seconds
TEPROLIN_RETRIES = 3


def _teprolin_process(text: str) -> List[Sentence]:
    """
    Send text to the local Teprolin Docker API (http://localhost:5000/process)
    and parse the response into a list of Sentence objects.

    Teprolin expects form-encoded POST data and returns JSON with structure:
      {
        "teprolin-result": {
          "tokenized": [[{token}, ...], ...]   # list of sentences
        }
      }

    Token fields: _id, _wordform, _lemma, _ctg (POS), _deprel, _head, _ner
    """
    for attempt in range(1, TEPROLIN_RETRIES + 1):
        try:
            exec_ops = "tokenization,sentence-splitting,pos-tagging,lemmatization,dependency-parsing"
            # NER is excluded — broken in this Docker build; NER is handled by spaCy fallback.
            # Send as raw form body — requests.utils.quote preserves commas in exec,
            # matching what curl -d sends (Teprolin rejects URL-encoded %2C commas).
            body = f"text={requests.utils.quote(text, safe='')}&exec={exec_ops}"
            resp = requests.post(
                TEPROLIN_URL,
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=TEPROLIN_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Teprolin attempt %d/%d failed: %s", attempt, TEPROLIN_RETRIES, exc)
            if attempt == TEPROLIN_RETRIES:
                raise
            time.sleep(2 ** attempt)

    sentences: List[Sentence] = []
    result = data.get("teprolin-result", {})

    # Response structure: {"tokenized": [[{tok}, ...], ...]}
    tokenized = result.get("tokenized", []) if isinstance(result, dict) else []

    for sent_idx, raw_sent in enumerate(tokenized):
        tokens: List[Token] = []
        sent_text_parts = []

        for tok_data in raw_sent:
            idx   = int(tok_data.get("_id", len(tokens) + 1))
            wform = tok_data.get("_wordform", "")
            lemma = tok_data.get("_lemma", wform)
            pos   = tok_data.get("_ctg", "X")
            dep   = tok_data.get("_deprel", "")
            head  = int(tok_data.get("_head", 0))
            ner   = tok_data.get("_ner", "O") or "O"

            tokens.append(Token(
                index=idx,
                text=wform,
                lemma=lemma,
                pos=pos,
                dep=dep,
                head=head,
                ner=ner,
            ))
            sent_text_parts.append(wform)

        sent_text = " ".join(sent_text_parts)
        sentences.append(Sentence(index=sent_idx, text=sent_text, tokens=tokens))

    return sentences


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("ro_core_news_lg")
    return _spacy_nlp


# Map spaCy NER labels to simplified BIO tags
_SPACY_NER_MAP = {
    "PER": "B-PER", "PERSON": "B-PER",
    "ORG": "B-ORG",
    "GPE": "B-LOC", "LOC": "B-LOC",
    "DATE": "B-DATE", "TIME": "B-TIME",
    "MONEY": "B-MONEY", "CARDINAL": "B-NUM", "PERCENT": "B-NUM",
}


def _spacy_process(text: str) -> List[Sentence]:
    nlp = _get_spacy()
    doc = nlp(text)
    sentences: List[Sentence] = []

    # Build token-level NER lookup from doc.ents
    ner_map: dict[int, str] = {}
    for ent in doc.ents:
        for i, tok in enumerate(ent):
            prefix = "B-" if i == 0 else "I-"
            label = ent.label_
            ner_map[tok.i] = f"{prefix}{label}"

    for sent_idx, sent in enumerate(doc.sents):
        tokens: List[Token] = []
        for local_idx, tok in enumerate(sent):
            # Dependency head index within the sentence (1-based)
            if tok.head == tok:
                head_idx = 0  # root
            else:
                head_idx = tok.head.i - sent.start + 1

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
        stanza_kwargs = {"lang": "ro", "verbose": False, "use_gpu": False}
        if os.path.isdir(STANZA_DIR):
            stanza_kwargs["dir"] = STANZA_DIR
        _stanza_nlp = stanza.Pipeline(**stanza_kwargs)
    return _stanza_nlp


def _stanza_process(text: str) -> List[Sentence]:
    nlp = _get_stanza()
    doc = nlp(text)
    sentences: List[Sentence] = []

    for sent_idx, sent in enumerate(doc.sentences):
        tokens: List[Token] = []

        # Build NER lookup: word index (1-based) → BIO tag
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

class RomanianPreprocessor:
    """
    Main entry point for Romanian text preprocessing.

    Parameters
    ----------
    backend : str
        One of 'teprolin', 'spacy', 'stanza', or 'auto'.
        'auto' (default) tries Teprolin first, falls back to spaCy.
    """

    def __init__(self, backend: str = "auto"):
        assert backend in {"auto", "teprolin", "spacy", "stanza"}, \
            f"Unknown backend '{backend}'. Choose from: auto, teprolin, spacy, stanza"
        self.backend = backend

    def process(self, text: str) -> List[Sentence]:
        """
        Preprocess a Romanian text string and return annotated sentences.

        Parameters
        ----------
        text : str
            Raw Romanian article text.

        Returns
        -------
        List[Sentence]
            Annotated sentences with tokens (text, lemma, POS, dep, NER).
        """
        text = text.strip()
        if not text:
            return []

        if self.backend == "teprolin":
            return _teprolin_process(text)
        elif self.backend == "spacy":
            return _spacy_process(text)
        elif self.backend == "stanza":
            return _stanza_process(text)
        else:  # auto
            try:
                result = _teprolin_process(text)
                if result:
                    return result
            except Exception as exc:
                logger.warning("Teprolin failed, falling back to spaCy: %s", exc)
            return _spacy_process(text)

    def process_with_all(self, text: str) -> dict:
        """
        Run all three backends and return their outputs side-by-side.
        Useful for cross-validation and debugging during development.

        Returns
        -------
        dict with keys 'teprolin', 'spacy', 'stanza'
        """
        results = {}
        for name, fn in [("spacy", _spacy_process), ("stanza", _stanza_process)]:
            try:
                results[name] = fn(text)
            except Exception as exc:
                logger.error("%s failed: %s", name, exc)
                results[name] = []
        try:
            results["teprolin"] = _teprolin_process(text)
        except Exception as exc:
            logger.warning("Teprolin failed: %s", exc)
            results["teprolin"] = []
        return results

