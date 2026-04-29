"""
smoke_test.py
-------------
Verifies that every tool and resource required by the Romanian and Spanish
pipelines is correctly installed and accessible.

Usage:
    python scripts/smoke_test.py              # all checks
    python scripts/smoke_test.py --romanian   # Romanian checks only
    python scripts/smoke_test.py --spanish    # Spanish checks only

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

STANZA_DIR = os.path.join(PROJECT_ROOT, "models", "stanza")
STANZA_ES_DIR = os.path.join(PROJECT_ROOT, "models", "stanza_es")
BERT_CACHE = os.path.join(PROJECT_ROOT, "models", "bert_romanian")
BERT_MODEL_ID = "xlm-roberta-base"

SENTENCE_RO = "Guvernul a anunțat că economia a crescut cu 3% în primul trimestru al anului 2024."
SENTENCE_ES = "El gobierno anunció que la economía creció un 3 % en el primer trimestre de 2024."

PASS = "✔"
FAIL = "✗"

results = []


def check(name):
    """Decorator that wraps a check function and records pass/fail."""
    def decorator(fn):
        def wrapper():
            try:
                detail = fn()
                results.append((PASS, name, detail))
                print(f"  {PASS}  {name}: {detail}")
            except Exception as exc:
                results.append((FAIL, name, str(exc)))
                print(f"  {FAIL}  {name}: {exc}")
        return wrapper
    return decorator


# ===========================================================================
# Romanian checks
# ===========================================================================

# ---------------------------------------------------------------------------
# Check R1 — spaCy Romanian model
# ---------------------------------------------------------------------------
@check("spaCy ro_core_news_lg")
def test_spacy():
    import spacy
    nlp = spacy.load("ro_core_news_lg")
    doc = nlp(SENTENCE_RO)
    tokens = [(t.text, t.pos_, t.dep_) for t in doc]
    return f"{len(tokens)} tokens parsed, root='{doc[list(doc.sents)[0].root.i].text}'"


# ---------------------------------------------------------------------------
# Check R2 — Stanza Romanian model
# ---------------------------------------------------------------------------
@check("Stanza Romanian")
def test_stanza():
    import stanza
    if os.path.isdir(STANZA_DIR):
        nlp = stanza.Pipeline("ro", dir=STANZA_DIR, verbose=False)
    else:
        nlp = stanza.Pipeline("ro", verbose=False)
    doc = nlp(SENTENCE_RO)
    words = [(w.text, w.upos) for sent in doc.sentences for w in sent.words]
    return f"{len(words)} words, first='{words[0]}'"


# ---------------------------------------------------------------------------
# Check R3 — Teprolin REST API (local Docker at localhost:5000)
# ---------------------------------------------------------------------------
@check("Teprolin API (localhost:5000)")
def test_teprolin():
    import requests
    # Must send as form-encoded (not JSON). NER excluded — broken in this Docker build.
    r = requests.post(
        "http://localhost:5000/process",
        data={
            "text": "Guvernul a anuntat cresterea economica in 2024.",
            "exec": "tokenization,sentence-splitting,pos-tagging,lemmatization,dependency-parsing",
        },
        timeout=15,
    )
    r.raise_for_status()
    result = r.json().get("teprolin-result", {})
    if not isinstance(result, dict) or "tokenized" not in result:
        raise RuntimeError(f"Unexpected response: {str(result)[:100]}")
    n_tokens = sum(len(s) for s in result["tokenized"])
    sample = result["tokenized"][0][0]
    return f"HTTP 200, {n_tokens} tokens, sample lemma='{sample.get('_lemma','?')}' pos='{sample.get('_ctg','?')}'"


# ---------------------------------------------------------------------------
# Check R4 — XLM-RoBERTa (multilingual BERT, covers Romanian)
# ---------------------------------------------------------------------------
@check(f"XLM-RoBERTa ({BERT_MODEL_ID})")
def test_bert():
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_ID, cache_dir=BERT_CACHE)
    model = AutoModel.from_pretrained(BERT_MODEL_ID, cache_dir=BERT_CACHE)
    inputs = tokenizer(SENTENCE_RO, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    shape = tuple(outputs.last_hidden_state.shape)
    return f"vocab={tokenizer.vocab_size}, output shape={shape}"


# ---------------------------------------------------------------------------
# Check R5 — RoWordNet
# ---------------------------------------------------------------------------
@check("RoWordNet")
def test_rowordnet():
    import rowordnet
    wn = rowordnet.RoWordNet()
    synsets = wn.synsets(literal="bancă")
    total = len(list(wn.synsets()))
    return f"{total} total synsets; 'bancă' synsets: {synsets[:2]}"


# ---------------------------------------------------------------------------
# Check R6 — Romanian Wikipedia (wikimedia/wikipedia)
# ---------------------------------------------------------------------------
@check("Romanian Wikipedia (wikimedia)")
def test_wiki():
    from datasets import load_dataset
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ro",
        split="train[:5]",
        cache_dir=os.path.join(PROJECT_ROOT, "data", "raw", "wiki_ro"),
    )
    sample_title = ds[0]["title"]
    return f"{len(ds)} samples loaded, first article='{sample_title}'"


# ---------------------------------------------------------------------------
# Check R7 — FEVER Gold Evidence (copenlu/fever_gold_evidence)
# ---------------------------------------------------------------------------
@check("FEVER Gold Evidence (copenlu)")
def test_fever():
    from datasets import load_dataset
    ds = load_dataset(
        "copenlu/fever_gold_evidence",
        split="train[:5]",
        cache_dir=os.path.join(PROJECT_ROOT, "data", "raw", "fever"),
    )
    return f"{len(ds)} samples, columns={ds.column_names}"


# ---------------------------------------------------------------------------
# Check R8 — Romanian pipeline end-to-end
# ---------------------------------------------------------------------------
@check("Romanian Pipeline (end-to-end)")
def test_romanian_pipeline():
    from src.pipeline.pipeline import RomanianPipeline
    pipeline = RomanianPipeline(backend="spacy", use_wordnet=True, use_nli=False)
    article = (
        "Economia a crescut semnificativ în ultimul trimestru, conform datelor INS. "
        "Analiștii susțin că economia a scăzut în același interval de timp."
    )
    result = pipeline.run(article)
    return (
        f"sentences={result.num_sentences}, claims={result.num_claims}, "
        f"contradictions={len(result.contradictions)}"
    )


# ===========================================================================
# Spanish checks
# ===========================================================================

# ---------------------------------------------------------------------------
# Check ES1 — spaCy Spanish model
# ---------------------------------------------------------------------------
@check("spaCy es_core_news_lg (Spanish)")
def test_spacy_es():
    import spacy
    # Try large → medium → small
    for model_name in ("es_core_news_lg", "es_core_news_md", "es_core_news_sm"):
        try:
            nlp = spacy.load(model_name)
            doc = nlp(SENTENCE_ES)
            tokens = [(t.text, t.pos_, t.dep_) for t in doc]
            return (
                f"Model '{model_name}', "
                f"{len(tokens)} tokens, "
                f"root='{doc[list(doc.sents)[0].root.i].text}'"
            )
        except OSError:
            continue
    raise RuntimeError(
        "No Spanish spaCy model found. "
        "Run: python -m spacy download es_core_news_lg"
    )


# ---------------------------------------------------------------------------
# Check ES2 — Stanza Spanish model
# ---------------------------------------------------------------------------
@check("Stanza Spanish")
def test_stanza_es():
    import stanza
    kwargs = {"lang": "es", "verbose": False, "use_gpu": False}
    if os.path.isdir(STANZA_ES_DIR):
        kwargs["dir"] = STANZA_ES_DIR
    nlp = stanza.Pipeline(**kwargs)
    doc = nlp(SENTENCE_ES)
    words = [(w.text, w.upos) for sent in doc.sentences for w in sent.words]
    return f"{len(words)} words, first='{words[0]}'"


# ---------------------------------------------------------------------------
# Check ES3 — NLTK WordNet + OMW-1.4 (Spanish)
# ---------------------------------------------------------------------------
@check("NLTK WordNet + OMW-1.4 (Spanish 'spa')")
def test_nltk_wordnet_es():
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets("economía", lang="spa")
    if not synsets:
        raise RuntimeError(
            "No Spanish synsets found — run: "
            "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
        )
    lemmas = [l.name() for s in synsets[:2] for l in s.lemmas(lang="spa")]
    return f"{len(synsets)} synsets for 'economía'; sample lemmas: {lemmas[:5]}"


# ---------------------------------------------------------------------------
# Check ES4 — Spanish pipeline end-to-end
# ---------------------------------------------------------------------------
@check("Spanish Pipeline (end-to-end)")
def test_spanish_pipeline():
    from src.spanish_pipeline.pipeline import SpanishPipeline
    pipeline = SpanishPipeline(backend="spacy", use_wordnet=True, use_nli=False)
    article = (
        "La economía creció significativamente en el último trimestre, según datos del INE. "
        "Los analistas sostienen que la economía se contrajo en el mismo período."
    )
    result = pipeline.run(article)
    return (
        f"sentences={result.num_sentences}, claims={result.num_claims}, "
        f"contradictions={len(result.contradictions)}"
    )


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test all pipeline dependencies."
    )
    parser.add_argument("--romanian", action="store_true",
                        help="Run Romanian checks only.")
    parser.add_argument("--spanish", action="store_true",
                        help="Run Spanish checks only.")
    args = parser.parse_args()

    run_romanian = not args.spanish  # run unless --spanish-only
    run_spanish  = not args.romanian  # run unless --romanian-only

    romanian_checks = [
        test_spacy,
        test_stanza,
        test_teprolin,
        test_bert,
        test_rowordnet,
        test_wiki,
        test_fever,
        test_romanian_pipeline,
    ]

    spanish_checks = [
        test_spacy_es,
        test_stanza_es,
        test_nltk_wordnet_es,
        test_spanish_pipeline,
    ]

    checks = []
    if run_romanian:
        checks.extend(romanian_checks)
    if run_spanish:
        checks.extend(spanish_checks)

    print("=" * 65)
    if run_romanian and run_spanish:
        print("  Romanian + Spanish Pipeline — Smoke Test")
    elif run_romanian:
        print("  Romanian Pipeline — Smoke Test")
    else:
        print("  Spanish Pipeline — Smoke Test")
    print("=" * 65)

    if run_romanian:
        print("\n--- Romanian checks ---")
        for fn in romanian_checks:
            fn()

    if run_spanish:
        print("\n--- Spanish checks ---")
        for fn in spanish_checks:
            fn()

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    print(f"  Results: {passed}/{len(results)} passed, {failed} failed")

    if failed:
        print("\n  Failed checks:")
        for icon, name, detail in results:
            if icon == FAIL:
                print(f"    - {name}: {detail}")
        print("=" * 65)
        sys.exit(1)
    else:
        print("  All checks passed.")
        print("=" * 65)
        sys.exit(0)


if __name__ == "__main__":
    main()

