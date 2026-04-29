# News Inconsistency Detector — Romanian & Spanish

Detects internal contradictions within Romanian and Spanish news articles using
a multi-layer rule-based + NLI pipeline.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - [Step 1 — Clone & create virtual environment](#step-1--clone--create-virtual-environment)
   - [Step 2 — Install Python dependencies](#step-2--install-python-dependencies)
   - [Step 3 — Install spaCy language models](#step-3--install-spacy-language-models)
   - [Step 4 — Download all models & datasets](#step-4--download-all-models--datasets)
   - [Step 5 — Verify the installation](#step-5--verify-the-installation)
3. [Running the Streamlit interface](#running-the-streamlit-interface)
4. [Running the evaluation](#running-the-evaluation)
5. [Quick pipeline usage](#quick-pipeline-usage)
6. [Project structure](#project-structure)
7. [Pipeline architecture](#pipeline-architecture)
8. [Datasets used](#datasets-used)
9. [Optional — Teprolin backend (Docker)](#optional--teprolin-backend-docker)
10. [Team](#team)

---

## Requirements

| Requirement | Version / Notes |
|---|---|
| Python | 3.10 or 3.11 recommended |
| pip | ≥ 23 |
| RAM | ≥ 8 GB (16 GB recommended when NLI is enabled) |
| Disk | ≥ 5 GB free (models + datasets) |
| Internet | Required for first-time model/dataset downloads |

> **GPU is not required.** All models run on CPU. Enabling NLI (`--use-nli`) is slower on CPU (~1–2 s per sentence pair).

---

## Installation

### Step 1 — Clone & create virtual environment

```bash
git clone <repo-url> NewsInconsistencies
cd NewsInconsistencies

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs all pinned packages including:
- `spacy 3.8` — NLP preprocessing (Romanian + Spanish)
- `stanza 1.11` — Stanford NLP (Romanian + Spanish)
- `transformers 5.x` + `torch (CPU)` — XLM-RoBERTa embeddings and NLI
- `datasets 4.x` — HuggingFace dataset loader
- `rowordnet 1.1` — Romanian WordNet
- `nltk ≥ 3.8` — NLTK (Open Multilingual WordNet for Spanish)
- `streamlit 1.56` — web interface
- `scikit-learn`, `numpy`, `pandas` — evaluation utilities

> **PyTorch CPU wheels** (`torch==2.11.0+cpu`) are specified in `requirements.txt`.  
> If you need GPU support, install PyTorch separately following https://pytorch.org/get-started/locally/.

### Step 3 — Install spaCy language models

The spaCy models are large binaries distributed separately from `requirements.txt`.

**Romanian (required):**
```bash
python -m spacy download ro_core_news_lg
```

**Spanish (required for the Spanish pipeline):**
```bash
python -m spacy download es_core_news_lg
```

> If disk space is limited you can use the smaller `es_core_news_md` or `es_core_news_sm` — the pipeline auto-falls-back to them.

### Step 4 — Download all models & datasets

Run the setup script once after installation. It downloads and caches all remaining resources:

```bash
python scripts/setup_downloads.py
```

**What it downloads (≈ 2–3 GB total):**

| Step | Resource | Size |
|---|---|---|
| RO-1 | Stanza Romanian model (tokeniser, POS, lemma, depparse) | ~200 MB |
| RO-2 | RoWordNet XML (Romanian WordNet) | ~20 MB |
| RO-3 | Romanian Wikipedia (`wikimedia/wikipedia 20231101.ro`) | ~1.5 GB (cached) |
| RO-3 | FEVER Gold Evidence (`copenlu/fever_gold_evidence`) | ~50 MB |
| RO-4 | XLM-RoBERTa base (`xlm-roberta-base`) — sentence embeddings | ~1.1 GB |
| RO-5 | Multilingual NLI model (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`) | ~120 MB |
| ES-1 | spaCy `es_core_news_lg` (if not already installed in Step 3) | ~568 MB |
| ES-2 | Stanza Spanish model | ~200 MB |
| ES-3 | NLTK WordNet + Open Multilingual Wordnet (`omw-1.4`) | ~50 MB |

All models are stored under `models/` and all datasets under `data/raw/` (both gitignored).

**Selective downloads** — if you only need one language or want to skip large downloads:

```bash
# Skip datasets (Romanian Wikipedia + FEVER) — saves ~1.5 GB
python scripts/setup_downloads.py --skip-datasets

# Skip NLI model (not needed unless --use-nli is passed)
python scripts/setup_downloads.py --skip-nli

# Skip BERT/XLM-RoBERTa (not needed unless --use-nli or embeddings are used)
python scripts/setup_downloads.py --skip-bert

# Spanish resources only (skip Romanian-specific downloads)
python scripts/setup_downloads.py --spanish-only

# Skip all Spanish resources
python scripts/setup_downloads.py --skip-spanish

# All flags combined
python scripts/setup_downloads.py --skip-datasets --skip-bert --skip-nli --skip-spanish
```

**NLTK data** (downloaded automatically by `setup_downloads.py`, but can also be run manually):
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Step 5 — Verify the installation

```bash
python scripts/smoke_test.py
```

Expected output (11/12 checks pass; Teprolin requires Docker — see [below](#optional--teprolin-backend-docker)):

```
=================================================================
  Romanian + Spanish Pipeline — Smoke Test
=================================================================

--- Romanian checks ---
  ✔  spaCy ro_core_news_lg: 16 tokens parsed, root='anunțat'
  ✔  Stanza Romanian: 16 words, first='('Guvernul', 'NOUN')'
  ✗  Teprolin API (localhost:5000): Connection refused   ← optional
  ✔  XLM-RoBERTa (xlm-roberta-base): vocab=250002, output shape=(1, 19, 768)
  ✔  RoWordNet: 59348 total synsets; 'bancă' synsets: [...]
  ✔  Romanian Wikipedia (wikimedia): 5 samples loaded
  ✔  FEVER Gold Evidence (copenlu): 5 samples, columns=[...]
  ✔  Romanian Pipeline (end-to-end): sentences=2, claims=3, contradictions=1

--- Spanish checks ---
  ✔  spaCy es_core_news_lg (Spanish): Model 'es_core_news_lg', 17 tokens
  ✔  Stanza Spanish: 17 words, first='('El', 'DET')'
  ✔  NLTK WordNet + OMW-1.4 (Spanish 'spa'): 3 synsets for 'economía'
  ✔  Spanish Pipeline (end-to-end): sentences=2, claims=3, contradictions=0

=================================================================
  Results: 11/12 passed, 1 failed
=================================================================
```

You can also run language-specific checks only:

```bash
python scripts/smoke_test.py --romanian   # Romanian checks only
python scripts/smoke_test.py --spanish    # Spanish checks only
```

---

## Running the Streamlit interface

```bash
streamlit run app.py
```

Opens at http://localhost:8501. The interface has two language tabs:

- **🇷🇴 Romanian Article** — analyse a Romanian news article with selectable backend (`auto` / `spacy` / `stanza` / `teprolin`)
- **🇪🇸 Spanish Article** — analyse a Spanish news article with selectable backend (`auto` / `spacy` / `stanza`)

Both tabs offer sample articles, a full contradiction breakdown, and JSON export.

---

## Running the evaluation

```bash
# Evaluate the Romanian pipeline (default — 100 injected + 50 real samples)
python scripts/run_evaluation.py

# Evaluate the Spanish pipeline
python scripts/run_evaluation.py --language spanish

# Evaluate both pipelines
python scripts/run_evaluation.py --language both

# Smaller quick run
python scripts/run_evaluation.py --language both --n-injected 20 --n-real 10

# Enable NLI layer (slower)
python scripts/run_evaluation.py --language both --use-nli

# Change preprocessing backend
python scripts/run_evaluation.py --backend stanza          # Romanian
python scripts/run_evaluation.py --es-backend stanza       # Spanish
```

Reports are saved to:
- `data/processed/evaluation/eval_report.json` — Romanian metrics
- `data/processed/evaluation/eval_report_es.json` — Spanish metrics

**Latest benchmark results (spaCy backend, no NLI):**

| Pipeline | Precision | Recall | F1 |
|---|---|---|---|
| Romanian (30 samples) | 0.833 | 0.750 | 0.789 |
| Spanish (10 samples) | 1.000 | 0.500 | 0.667 |

---

## Quick pipeline usage

**Romanian:**
```python
from src.pipeline.pipeline import RomanianPipeline

pipeline = RomanianPipeline(backend="spacy")   # 'auto' | 'spacy' | 'stanza' | 'teprolin'
result = pipeline.run(
    "Compania a raportat un profit de 5 milioane de euro în primul trimestru. "
    "Directorul a declarat că profitul a crescut la 12 milioane de euro în primul trimestru."
)
print(result)          # PipelineResult(sentences=2, claims=2, contradictions=1)
print(result.to_json())
```

**Spanish:**
```python
from src.spanish_pipeline.pipeline import SpanishPipeline

pipeline = SpanishPipeline(backend="spacy")    # 'auto' | 'spacy' | 'stanza'
result = pipeline.run(
    "El gobierno anunció que la economía creció un 3 % en 2023. "
    "Según otro raport, economia s-a contractat cu 5 % în 2023."
)
print(result)
print(result.to_json())
```

**With NLI (both languages):**
```python
pipeline = RomanianPipeline(use_nli=True, nli_threshold=0.65)
pipeline = SpanishPipeline(use_nli=True, nli_threshold=0.65)
```

---

## Project structure

```
NewsInconsistencies/
├── app.py                              # Streamlit web interface (RO + ES tabs)
├── requirements.txt                    # Pinned Python dependencies
├── scripts/
│   ├── setup_downloads.py              # Download all models & datasets (run once)
│   ├── smoke_test.py                   # Verify all dependencies work
│   └── run_evaluation.py               # Benchmark evaluation (RO + ES)
├── src/
│   ├── pipeline/                       # Romanian pipeline
│   │   ├── preprocessor.py             # Teprolin / spaCy / Stanza backend
│   │   ├── claim_extractor.py          # SPO triplet extraction
│   │   ├── normalizer.py               # Numeric & temporal normalisation
│   │   ├── wordnet_checker.py          # RoWordNet antonymy detection
│   │   ├── contradiction_detector.py   # Rule-based + NLI detection
│   │   ├── embeddings.py               # XLM-RoBERTa sentence embeddings
│   │   ├── nli_module.py               # NLI inference wrapper
│   │   ├── evaluator.py                # BenchmarkBuilder + Evaluator
│   │   └── pipeline.py                 # RomanianPipeline orchestrator
│   └── spanish_pipeline/               # Spanish pipeline (mirrors Romanian)
│       ├── preprocessor.py             # spaCy / Stanza backend (Spanish)
│       ├── claim_extractor.py          # Spanish SPO + numeric/temporal patterns
│       ├── normalizer.py               # Spanish unit & date normalisation
│       ├── wordnet_checker.py          # NLTK OMW Spanish antonymy detection
│       ├── contradiction_detector.py   # Rule-based + NLI detection (Spanish)
│       └── pipeline.py                 # SpanishPipeline orchestrator
├── data/
│   ├── raw/                            # Downloaded datasets (gitignored)
│   ├── embeddings/                     # Cached embeddings (gitignored)
│   └── processed/evaluation/           # Benchmark & eval reports
└── models/
    ├── bert_romanian/                  # XLM-RoBERTa base cache (gitignored)
    ├── nli/                            # NLI model cache (gitignored)
    ├── stanza/                         # Stanza Romanian models (gitignored)
    └── stanza_es/                      # Stanza Spanish models (gitignored)
```

---

## Pipeline architecture

Both pipelines share the same four-phase architecture:

```
Article text
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1 — Preprocessor                │
│  Sentence splitting · POS tagging       │
│  Lemmatisation · Dependency parsing     │
│  Backends: spaCy | Stanza | Teprolin*  │
└──────────────────┬──────────────────────┘
                   │ Sentence + Token objects
                   ▼
┌─────────────────────────────────────────┐
│  Phase 2 — Claim Extractor              │
│  SPO triplet extraction                 │
│  Numeric / temporal value detection     │
└──────────────────┬──────────────────────┘
                   │ Claim objects
                   ▼
┌─────────────────────────────────────────┐
│  Phase 3 — Normalizer                   │
│  Unit canonicalisation (€, %, km…)      │
│  Date → ISO 8601                        │
└──────────────────┬──────────────────────┘
                   │ Normalised claim data
                   ▼
┌─────────────────────────────────────────┐
│  Phase 4 — Contradiction Detector       │
│  Layer 1: Numeric rules                 │
│  Layer 2: Temporal rules                │
│  Layer 3: Entity rules                  │
│  Layer 4: WordNet antonymy/neg-synonym  │
│  Layer 5: NLI (opt.) XLM-RoBERTa MiniLM│
└──────────────────┬──────────────────────┘
                   │ ContradictionAlert list
                   ▼
             PipelineResult
```

\* Teprolin is Romanian-only and requires Docker (see below).

---

## Datasets used

| Dataset | HuggingFace ID | Language | Use |
|---|---|---|---|
| Romanian Wikipedia | `wikimedia/wikipedia` `20231101.ro` | Romanian | Factual claim injection |
| FEVER Gold Evidence | `copenlu/fever_gold_evidence` | English | Evaluation reference |
| MLSUM Romanian | `mlsum` / `ro` | Romanian | (optional) working set |
| XLSum Romanian | `csebuetnlp/xlsum` / `romanian` | Romanian | Evaluation benchmark |
| XNLI Romanian | `xnli` / `ro` | Romanian | Contradiction label reference |

All datasets are open access and require no registration.

---

## Optional — Teprolin backend (Docker)

Teprolin is a Romanian-language NLP REST API. It provides higher-quality lemmatisation
and POS tagging for Romanian but requires Docker.

```bash
# Pull and start the Teprolin container
docker pull racai/teprolin
docker run -d -p 5000:5000 racai/teprolin

# Use the teprolin backend
python scripts/run_evaluation.py --backend teprolin
```

Once the container is running, the smoke test's `Teprolin API` check will also pass.
Without Docker, the pipeline falls back to spaCy (`backend="auto"` → tries spaCy first).

---

## Team

Bejinaru-Manoila Angel, Bocancea Georgiana-Letitia, Cojocaru Paul-Cristian, Herce Claudia
