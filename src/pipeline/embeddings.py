"""
embeddings.py
-------------
Phase 3 — Embedding-based semantic similarity for the Romanian pipeline.

Provides two services:
  1. Static embeddings (fastText / word2vec via gensim):
     - Word-level similarity for predicate and entity comparison
     - OOV handling via fastText subword representations

  2. Contextual embeddings (Romanian BERT via Hugging Face):
     - Sentence-level embeddings via mean pooling of the last hidden state
     - Used for claim-pair similarity scoring and soft co-reference

Both are used to enrich the claim relatedness signal in the contradiction
detector, particularly for cases where lemma-exact matching fails.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BERT_CACHE = os.path.join(PROJECT_ROOT, "models", "bert_romanian")

# ---------------------------------------------------------------------------
# Static embeddings — fastText / word2vec via gensim
# ---------------------------------------------------------------------------

_static_model = None
_static_model_path: Optional[str] = None


def load_static_embeddings(path: str, binary: bool = True):
    """
    Load a pre-trained static embedding model (fastText .bin or word2vec .bin/.txt).

    Parameters
    ----------
    path : str
        Absolute path to the model file (e.g. 'data/embeddings/cc.ro.300.bin').
    binary : bool
        True for binary word2vec/fastText format, False for plain text GloVe.
    """
    global _static_model, _static_model_path
    if _static_model is not None and _static_model_path == path:
        return _static_model

    logger.info("Loading static embeddings from %s…", path)
    try:
        from gensim.models import KeyedVectors
        from gensim.models.fasttext import load_facebook_vectors
    except ImportError:
        raise ImportError("gensim is required: pip install gensim")

    if path.endswith(".bin") and "fasttext" in path.lower() or "cc." in path.lower():
        _static_model = load_facebook_vectors(path)
    else:
        _static_model = KeyedVectors.load_word2vec_format(path, binary=binary)

    _static_model_path = path
    logger.info("Static embeddings loaded — vocab size: %d", len(_static_model))
    return _static_model


def word_similarity(word_a: str, word_b: str) -> float:
    """
    Return cosine similarity [0,1] between two words using static embeddings.
    Returns 0.0 if either word is OOV and no subword model is available.
    """
    if _static_model is None:
        logger.debug("Static embeddings not loaded — returning 0.0 for similarity.")
        return 0.0
    try:
        return float(_static_model.similarity(word_a.lower(), word_b.lower()))
    except KeyError:
        return 0.0


def sentence_vector_static(text: str) -> Optional[np.ndarray]:
    """
    Compute a mean word vector for a sentence using static embeddings.
    Returns None if the model is not loaded.
    """
    if _static_model is None:
        return None
    words = text.lower().split()
    vecs = []
    for w in words:
        try:
            vecs.append(_static_model[w])
        except KeyError:
            pass
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


# ---------------------------------------------------------------------------
# Contextual embeddings — Romanian BERT
# ---------------------------------------------------------------------------

_bert_tokenizer = None
_bert_model = None
_BERT_MODEL_ID = "xlm-roberta-base"


def _get_bert():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading Romanian BERT…")
        _bert_tokenizer = AutoTokenizer.from_pretrained(
            _BERT_MODEL_ID, cache_dir=BERT_CACHE
        )
        _bert_model = AutoModel.from_pretrained(
            _BERT_MODEL_ID, cache_dir=BERT_CACHE
        )
        _bert_model.eval()
        logger.info("Romanian BERT loaded.")
    return _bert_tokenizer, _bert_model


def sentence_embedding_bert(text: str) -> np.ndarray:
    """
    Compute a sentence embedding using Romanian BERT (mean pooling of last hidden state).

    Parameters
    ----------
    text : str

    Returns
    -------
    np.ndarray of shape (768,)
    """
    import torch
    tokenizer, model = _get_bert()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool over token dimension (exclude [CLS] and [SEP] for cleaner reps)
    hidden = outputs.last_hidden_state  # (1, seq_len, 768)
    attention_mask = inputs["attention_mask"]  # (1, seq_len)
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    pooled = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
    return pooled.squeeze(0).numpy()


@lru_cache(maxsize=512)
def _cached_sentence_embedding_bert(text: str) -> np.ndarray:
    return sentence_embedding_bert(text)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# High-level services used by the pipeline
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Provides embedding-based similarity scores for claim pairs.

    Falls back gracefully if models are not loaded:
      - Static model not loaded → word_similarity returns 0.0
      - BERT not loaded → bert_similarity raises, caught externally

    Parameters
    ----------
    use_bert : bool
        If True, load Romanian BERT for sentence-level similarity (slower, ~500 MB).
    static_model_path : str or None
        Path to fastText/word2vec .bin file. If None, static similarity is disabled.
    """

    def __init__(self, use_bert: bool = False, static_model_path: Optional[str] = None):
        self.use_bert = use_bert
        self.static_model_path = static_model_path
        if static_model_path and os.path.isfile(static_model_path):
            load_static_embeddings(static_model_path)
        elif static_model_path:
            logger.warning(
                "Static embedding file not found: %s — word similarity disabled.", static_model_path
            )

    def predicate_similarity(self, lemma_a: str, lemma_b: str) -> float:
        """
        Return semantic similarity between two predicate lemmas.
        Uses static embeddings if available, else 1.0 (exact) / 0.0 (different).
        """
        if lemma_a == lemma_b:
            return 1.0
        if _static_model is not None:
            return word_similarity(lemma_a, lemma_b)
        return 0.0

    def claim_similarity_static(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between two claim signatures using static embeddings."""
        vec_a = sentence_vector_static(text_a)
        vec_b = sentence_vector_static(text_b)
        if vec_a is None or vec_b is None:
            return 0.0
        return cosine_similarity(vec_a, vec_b)

    def claim_similarity_bert(self, text_a: str, text_b: str) -> float:
        """
        Return sentence-level cosine similarity between two claim texts using BERT.
        Loads BERT on first call (lazy).
        """
        try:
            vec_a = _cached_sentence_embedding_bert(text_a)
            vec_b = _cached_sentence_embedding_bert(text_b)
            return cosine_similarity(vec_a, vec_b)
        except Exception as exc:
            logger.warning("BERT similarity failed: %s", exc)
            return 0.0

    @staticmethod
    def _claim_text(claim) -> str:
        parts = [claim.subject, claim.predicate, claim.object]
        qualifiers = getattr(claim, "qualifiers", None) or []
        parts.extend(qualifiers)
        return " ".join(part.strip() for part in parts if part and part.strip())

    def claims_are_semantically_related(
        self,
        claim_a,
        claim_b,
        predicate_threshold: float = 0.75,
        sentence_threshold: float = 0.80,
    ) -> bool:
        """
        Return True if two claims are semantically related enough to compare for contradictions.

        Uses predicate similarity (static) and optionally sentence similarity (BERT).
        This complements the lemma-based `_claims_are_related` check in the detector.
        """
        la = (claim_a.predicate_lemma or "").lower()
        lb = (claim_b.predicate_lemma or "").lower()

        pred_sim = self.predicate_similarity(la, lb)
        if pred_sim >= predicate_threshold:
            return True

        text_a = self._claim_text(claim_a)
        text_b = self._claim_text(claim_b)
        if not text_a or not text_b:
            return False

        if _static_model is not None:
            static_sim = self.claim_similarity_static(text_a, text_b)
            if static_sim >= sentence_threshold:
                return True

        if self.use_bert:
            sent_sim = self.claim_similarity_bert(text_a, text_b)
            if sent_sim >= sentence_threshold:
                return True

        return False

