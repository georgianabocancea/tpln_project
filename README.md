# Romanian News Inconsistency Detector

Detection of internal contradictions within Romanian news articles.

## Project Structure

```
NewsInconsistencies/
├── app.py                          # Streamlit interface
├── requirements.txt                # Pinned dependencies
├── scripts/
│   ├── setup_downloads.py          # Download models & datasets (run once)
│   └── smoke_test.py               # Verify all tools work
├── src/
│   └── pipeline/
│       ├── preprocessor.py         # Teprolin / spaCy / Stanza preprocessing
│       ├── claim_extractor.py      # SPO triplet extraction
│       ├── normalizer.py           # Numeric & temporal normalisation
│       ├── contradiction_detector.py  # Rule-based + NLI detection
│       └── pipeline.py             # End-to-end orchestrator
├── data/
│   ├── raw/                        # Downloaded datasets (gitignored)
│   ├── embeddings/                 # Word vectors (gitignored)
│   └── processed/                  # Pipeline outputs
├── models/
│   ├── bert_romanian/              # Cached Romanian BERT (gitignored)
│   └── stanza/                     # Stanza Romanian model (gitignored)
└── notebooks/                      # Exploratory analysis
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download ro_core_news_lg
```

### 2. Download models and datasets
```bash
python scripts/setup_downloads.py
```

This downloads:
- Stanza Romanian model (~200 MB)
- RoWordNet XML (~20 MB)
- MLSUM Romanian, XLSum Romanian, Romanian Wikipedia, XNLI Romanian (via HuggingFace)
- Romanian BERT (`dumitrescuemilandrei/bert-base-romanian-cased-v1`)

### 3. Verify installation
```bash
python scripts/smoke_test.py
```

All 10 checks must pass before proceeding.

### 4. Run the interface
```bash
streamlit run app.py
```

## Quick pipeline usage

```python
from src.pipeline.pipeline import RomanianPipeline

pipeline = RomanianPipeline(backend="spacy")   # or 'auto', 'teprolin', 'stanza'
result = pipeline.run("""
    Compania a raportat un profit de 5 milioane de euro în primul trimestru.
    Directorul a declarat că profitul a crescut la 12 milioane de euro în primul trimestru.
""")

print(result)
print(result.to_json())
```

## Datasets used (open, no registration required)

| Dataset | Source | Use |
|---|---|---|
| MLSUM Romanian | `mlsum` / `ro` on HuggingFace | Working set (Digi24 news) |
| XLSum Romanian | `csebuetnlp/xlsum` / `romanian` | Evaluation benchmark |
| Romanian Wikipedia | `wikipedia` / `20231101.ro` | Factual claim injection |
| XNLI Romanian | `xnli` / `ro` | Contradiction label reference |

## Pipeline phases

| Phase | Status | Module |
|---|---|---|
| 1 — Environment & data prep | ✅ Done | `scripts/` |
| 2 — Claim extraction & normalisation | ✅ Done | `preprocessor.py`, `claim_extractor.py`, `normalizer.py` |
| 3 — Deterministic contradiction detection | ✅ Done | `contradiction_detector.py` (Layer 1) |
| 4 — NLI soft detection | 🔧 Stub ready | `contradiction_detector.py` (Layer 2, `use_nli=True`) |
| 5 — Evaluation benchmark | 🔜 Next | `scripts/evaluate.py` (TBD) |
| 6 — Interface & export | ✅ Done | `app.py` |

## Team

Bejinaru-Manoila Angel, Bocancea Georgiana-Letitia, Cojocaru Paul-Cristian, Herce Claudia

