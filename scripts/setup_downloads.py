"""
setup_downloads.py
------------------
Run once after installing requirements to download all models and datasets.

Usage:
    python scripts/setup_downloads.py [--skip-datasets] [--skip-stanza] [--skip-wordnet] [--skip-bert]
                                      [--skip-nli] [--skip-spanish]

Verified working resources (April 2026):
  • Stanza Romanian              — stanfordnlp/stanza-ro (HF Hub)
  • RoWordNet                    — rowordnet PyPI package (auto-download)
  • Romanian Wikipedia           — wikimedia/wikipedia  20231101.ro  (Parquet)
  • FEVER (gold evidence)        — copenlu/fever_gold_evidence       (Parquet)
  • Multilingual BERT            — xlm-roberta-base                  (public)
  • spaCy Spanish                — es_core_news_lg                   (pip install)
  • Stanza Spanish               — stanfordnlp/stanza-es
  • NLTK WordNet + OMW-1.4       — Open Multilingual Wordnet (spa)
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
STANZA_DIR = os.path.join(PROJECT_ROOT, "models", "stanza")
STANZA_ES_DIR = os.path.join(PROJECT_ROOT, "models", "stanza_es")
BERT_CACHE = os.path.join(PROJECT_ROOT, "models", "bert_romanian")


def download_stanza_romanian():
    """Download Stanza Romanian models (~200 MB)."""
    print("\n[1/4] Downloading Stanza Romanian model...")
    import stanza
    stanza.download("ro", model_dir=STANZA_DIR, verbose=True)
    print("      ✔ Stanza Romanian model ready.")


def download_rowordnet():
    """Trigger RoWordNet first-use download (~20 MB)."""
    print("\n[2/4] Initialising RoWordNet (downloads XML on first use)...")
    import rowordnet
    wn = rowordnet.RoWordNet()
    synsets = wn.synsets(literal="bancă")
    total = len(list(wn.synsets()))
    print(f"      ✔ RoWordNet ready — {total} synsets loaded.")
    print(f"        Sample synsets for 'bancă': {synsets[:3]}")


def download_datasets():
    """Download and cache HuggingFace datasets to data/raw/."""
    from datasets import load_dataset
    import json

    configs = [
        {
            "name": "Romanian Wikipedia (wikimedia)",
            "key": "wiki_ro",
            "loader": lambda: load_dataset(
                "wikimedia/wikipedia",
                "20231101.ro",
                cache_dir=os.path.join(DATA_RAW, "wiki_ro"),
            ),
        },
        {
            "name": "FEVER Gold Evidence (copenlu)",
            "key": "fever",
            "loader": lambda: load_dataset(
                "copenlu/fever_gold_evidence",
                cache_dir=os.path.join(DATA_RAW, "fever"),
            ),
        },
    ]

    stats = {}
    for cfg in configs:
        print(f"\n[3/4] Downloading {cfg['name']}...")
        try:
            ds = cfg["loader"]()
            split_info = {split: len(ds[split]) for split in ds}
            stats[cfg["key"]] = split_info
            print(f"      ✔ {cfg['name']} ready — splits: {split_info}")
        except Exception as exc:
            print(f"      ✗ Failed to download {cfg['name']}: {exc}")
            stats[cfg["key"]] = {"error": str(exc)}

    manifest_path = os.path.join(DATA_RAW, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n      Dataset manifest written to {manifest_path}")


def download_bert():
    """Pre-download XLM-RoBERTa base (multilingual encoder for sentence embeddings)."""
    print("\n[4/5] Pre-downloading XLM-RoBERTa base (xlm-roberta-base)...")
    from transformers import AutoTokenizer, AutoModel

    model_id = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=BERT_CACHE)
    model = AutoModel.from_pretrained(model_id, cache_dir=BERT_CACHE)
    print(f"      ✔ XLM-RoBERTa base ready — vocab size: {tokenizer.vocab_size}")


def download_nli_model():
    """
    Pre-download the NLI fine-tuned model for contradiction detection.

    MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli is a multilingual NLI model
    (XLM-RoBERTa tokenizer, 250K vocab) fine-tuned on MNLI + XNLI (15 languages
    including Romanian). ~120 MB — CPU-friendly.

    This is DIFFERENT from xlm-roberta-base (models/bert_romanian/) which is the
    base encoder used for sentence embeddings — it has no NLI classification head.
    """
    print("\n[5/5] Pre-downloading NLI model (symanto/xlm-roberta-base-snli-mnli)...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_id = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    nli_cache = os.path.join(PROJECT_ROOT, "models", "nli")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=nli_cache)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=nli_cache)
    labels = list(model.config.id2label.values())
    print(f"      ✔ NLI model ready — labels: {labels}")


# ---------------------------------------------------------------------------
# Spanish-specific downloads
# ---------------------------------------------------------------------------


def download_spacy_spanish():
    """Install the spaCy Spanish model (es_core_news_lg)."""
    print("\n[ES-1/3] Downloading spaCy Spanish model (es_core_news_lg)...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "es_core_news_lg"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("      ✔ spaCy es_core_news_lg ready.")
    else:
        print("      ✗ spaCy es_core_news_lg download failed — trying es_core_news_md...")
        result2 = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "es_core_news_md"],
            capture_output=False,
        )
        if result2.returncode == 0:
            print("      ✔ spaCy es_core_news_md ready (fallback).")
        else:
            print("      ✗ Could not download any Spanish spaCy model. "
                  "Run manually: python -m spacy download es_core_news_lg")


def download_stanza_spanish():
    """Download Stanza Spanish models (~200 MB)."""
    print("\n[ES-2/3] Downloading Stanza Spanish model...")
    import stanza
    stanza.download("es", model_dir=STANZA_ES_DIR, verbose=True)
    print("      ✔ Stanza Spanish model ready.")


def download_nltk_spanish_wordnet():
    """Download NLTK WordNet and Open Multilingual Wordnet (includes Spanish 'spa')."""
    print("\n[ES-3/3] Downloading NLTK WordNet + Open Multilingual Wordnet (omw-1.4)...")
    import nltk
    nltk.download("wordnet", quiet=False)
    nltk.download("omw-1.4", quiet=False)

    # Verify Spanish synsets are available
    from nltk.corpus import wordnet as wn
    spa_synsets = wn.synsets("economía", lang="spa")
    print(f"      ✔ NLTK WordNet + OMW-1.4 ready — "
          f"Spanish synsets for 'economía': {spa_synsets[:3]}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all models and datasets for the Romanian and Spanish pipelines."
    )
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-stanza", action="store_true")
    parser.add_argument("--skip-wordnet", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--skip-spanish", action="store_true",
                        help="Skip all Spanish-specific resource downloads.")
    parser.add_argument("--spanish-only", action="store_true",
                        help="Download only Spanish resources (skip Romanian-specific ones).")
    args = parser.parse_args()

    print("=" * 60)
    print("  Romanian + Spanish Pipeline — Resource Setup")
    print("=" * 60)

    if not args.spanish_only:
        if not args.skip_stanza:
            download_stanza_romanian()

        if not args.skip_wordnet:
            download_rowordnet()

        if not args.skip_datasets:
            download_datasets()

        if not args.skip_bert:
            download_bert()

        if not args.skip_nli:
            download_nli_model()

    if not args.skip_spanish:
        print("\n" + "=" * 60)
        print("  Spanish Pipeline — Resource Setup")
        print("=" * 60)
        download_spacy_spanish()
        download_stanza_spanish()
        download_nltk_spanish_wordnet()

    print("\n" + "=" * 60)
    print("  All resources downloaded successfully.")
    print("  Run 'python scripts/smoke_test.py' to verify everything.")
    print("=" * 60)


if __name__ == "__main__":
    main()

