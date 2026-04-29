"""
app.py
------
Streamlit interface for the Romanian and Spanish contradiction detection pipelines.

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json

from src.pipeline.pipeline import RomanianPipeline
from src.spanish_pipeline.pipeline import SpanishPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="News Inconsistency Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 News — Internal Consistency Checker")
st.caption(
    "Detects numeric, temporal, entity, and linguistic contradictions "
    "within a single news article (Romanian 🇷🇴 and Spanish 🇪🇸)."
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    backend_ro = st.selectbox(
        "Romanian preprocessing backend",
        options=["auto", "spacy", "stanza", "teprolin"],
        index=0,
        help=(
            "'auto' tries Teprolin first and falls back to spaCy. "
            "Teprolin requires internet access."
        ),
    )

    backend_es = st.selectbox(
        "Spanish preprocessing backend",
        options=["auto", "spacy", "stanza"],
        index=0,
        help="'auto' tries spaCy first and falls back to Stanza.",
    )

    use_wordnet = st.checkbox(
        "Enable WordNet checks (antonymy)",
        value=True,
        help="Uses RoWordNet (Romanian) / NLTK OMW-Spanish to detect antonym and negated-synonym contradictions.",
    )

    use_nli = st.checkbox(
        "Enable NLI soft detection (slower)",
        value=False,
        help="Uses XLM-RoBERTa multilingual NLI model to detect linguistic contradictions.",
    )

    nli_threshold = st.slider(
        "NLI confidence threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.65,
        step=0.05,
        disabled=not use_nli,
    )

    st.divider()
    st.markdown(
        "**Detection layers**\n"
        "- 🔴 **NUMERIC** — value conflicts (rule-based)\n"
        "- 🟠 **TEMPORAL** — date conflicts (rule-based)\n"
        "- 🟡 **ENTITY** — NE conflicts (rule-based)\n"
        "- 🔵 **LINGUISTIC** — WordNet antonymy / NLI\n"
    )
    st.divider()
    st.markdown(
        "**Romanian pipeline**\n"
        "- Teprolin / spaCy / Stanza — preprocessing\n"
        "- RoWordNet — antonymy detection\n"
        "- XLM-RoBERTa NLI — soft detection\n\n"
        "**Spanish pipeline**\n"
        "- spaCy `es_core_news_lg` / Stanza — preprocessing\n"
        "- NLTK Open Multilingual Wordnet (spa) — antonymy\n"
        "- XLM-RoBERTa NLI — soft detection\n"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ro, tab_es, tab_eval, tab_about = st.tabs(
    ["🇷🇴 Romanian Article", "🇪🇸 Spanish Article", "📊 Evaluation", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Helper — render contradiction results (shared by both language tabs)
# ---------------------------------------------------------------------------

BADGE_COLORS = {
    "NUMERIC":    "🔴",
    "TEMPORAL":   "🟠",
    "ENTITY":     "🟡",
    "LINGUISTIC": "🔵",
}


def render_pipeline_result(result):
    """Render metrics, contradiction alerts and export for any PipelineResult."""
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Sentences", result.num_sentences)
    m2.metric("Claims extracted", result.num_claims)
    m3.metric(
        "Contradictions found",
        len(result.contradictions),
        delta=None if len(result.contradictions) == 0 else f"{len(result.contradictions)} ⚠️",
        delta_color="inverse",
    )

    if result.contradictions:
        st.subheader("⚠️ Detected Contradictions")
        for i, alert in enumerate(result.contradictions, 1):
            badge = BADGE_COLORS.get(alert.contradiction_type, "⚪")
            with st.expander(
                f"{badge} Contradiction #{i} — {alert.contradiction_type} "
                f"(confidence: {alert.confidence:.0%})",
                expanded=True,
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Sentence A**")
                    st.info(alert.claim_a_text)
                    if alert.evidence_a:
                        st.caption(f"Conflicting fragment: *{alert.evidence_a}*")
                with col_b:
                    st.markdown("**Sentence B**")
                    st.info(alert.claim_b_text)
                    if alert.evidence_b:
                        st.caption(f"Conflicting fragment: *{alert.evidence_b}*")
                st.markdown(f"**Explanation:** {alert.explanation}")
    else:
        st.success("✅ No contradictions detected in this article.")

    st.divider()
    st.subheader("📤 Export Results")
    json_output = result.to_json()
    st.download_button(
        label="Download JSON",
        data=json_output,
        file_name="contradiction_report.json",
        mime="application/json",
    )
    with st.expander("View raw JSON"):
        st.code(json_output, language="json")


# ---------------------------------------------------------------------------
# Sample articles — Romanian
# ---------------------------------------------------------------------------

SAMPLE_ARTICLES_RO = {
    "— choose a sample —": "",
    "Numeric contradiction (injected)": (
        "Compania a raportat un profit de 5 milioane de euro în primul trimestru al anului 2024. "
        "Directorul general a declarat că profitul companiei a crescut la 12 milioane de euro "
        "în primul trimestru al anului 2024."
    ),
    "Temporal contradiction (injected)": (
        "Guvernul a aprobat bugetul în ședința din 15 martie 2023. "
        "Potrivit documentelor oficiale, bugetul a fost aprobat în ședința din septembrie 2024."
    ),
    "Antonym contradiction (WordNet)": (
        "Economia a crescut semnificativ în ultimul trimestru, conform datelor INS. "
        "Analiștii susțin că economia a scăzut în același interval de timp."
    ),
    "No contradiction": (
        "România a înregistrat o creștere economică de 3% în 2023. "
        "Banca Națională a României a menținut rata dobânzii de politică monetară la 7%."
    ),
}

# ---------------------------------------------------------------------------
# Sample articles — Spanish
# ---------------------------------------------------------------------------

SAMPLE_ARTICLES_ES = {
    "— choose a sample —": "",
    "Numeric contradiction (injected)": (
        "La empresa reportó ganancias de 5 millones de euros en el primer trimestre de 2024. "
        "El director general afirmó que las ganancias de la empresa ascendieron a 12 millones de euros "
        "en el primer trimestre de 2024."
    ),
    "Temporal contradiction (injected)": (
        "El gobierno aprobó el presupuesto en la sesión del 15 de marzo de 2023. "
        "Según los documentos oficiales, el presupuesto fue aprobado en la sesión de septiembre de 2024."
    ),
    "Antonym contradiction (WordNet)": (
        "La economía creció significativamente en el último trimestre, según datos del INE. "
        "Los analistas sostienen que la economía se contrajo en el mismo período."
    ),
    "No contradiction": (
        "España registró un crecimiento económico del 3% en 2023. "
        "El Banco de España mantuvo el tipo de interés de referencia en el 4%."
    ),
}

# ---------------------------------------------------------------------------
# Tab 1: Romanian Article
# ---------------------------------------------------------------------------

with tab_ro:
    col1, col2 = st.columns([2, 1])

    with col2:
        sample_key_ro = st.selectbox("Load sample article", list(SAMPLE_ARTICLES_RO.keys()))

    with col1:
        article_text_ro = st.text_area(
            "Article text (Romanian)",
            value=SAMPLE_ARTICLES_RO[sample_key_ro],
            height=250,
            placeholder="Paste your Romanian news article here…",
        )

    run_btn_ro = st.button("🚀 Analyse Romanian Article", type="primary", use_container_width=True, key="run_ro")

    if run_btn_ro:
        if not article_text_ro.strip():
            st.warning("Please enter or paste a Romanian article first.")
        else:
            with st.spinner("Running Romanian pipeline…"):
                try:
                    pipeline_ro = RomanianPipeline(
                        backend=backend_ro,
                        use_wordnet=use_wordnet,
                        use_nli=use_nli,
                        nli_threshold=nli_threshold,
                    )
                    result_ro = pipeline_ro.run(article_text_ro)
                except Exception as exc:
                    st.error(f"Romanian pipeline error: {exc}")
                    st.stop()

            render_pipeline_result(result_ro)

# ---------------------------------------------------------------------------
# Tab 2: Spanish Article
# ---------------------------------------------------------------------------

with tab_es:
    col1_es, col2_es = st.columns([2, 1])

    with col2_es:
        sample_key_es = st.selectbox("Load sample article", list(SAMPLE_ARTICLES_ES.keys()), key="sample_es")

    with col1_es:
        article_text_es = st.text_area(
            "Article text (Spanish)",
            value=SAMPLE_ARTICLES_ES[sample_key_es],
            height=250,
            placeholder="Paste your Spanish news article here…",
            key="article_es",
        )

    run_btn_es = st.button("🚀 Analyse Spanish Article", type="primary", use_container_width=True, key="run_es")

    if run_btn_es:
        if not article_text_es.strip():
            st.warning("Please enter or paste a Spanish article first.")
        else:
            with st.spinner("Running Spanish pipeline…"):
                try:
                    pipeline_es = SpanishPipeline(
                        backend=backend_es,
                        use_wordnet=use_wordnet,
                        use_nli=use_nli,
                        nli_threshold=nli_threshold,
                    )
                    result_es = pipeline_es.run(article_text_es)
                except Exception as exc:
                    st.error(f"Spanish pipeline error: {exc}")
                    st.stop()

            render_pipeline_result(result_es)

# ---------------------------------------------------------------------------
# Tab 3: Evaluation
# ---------------------------------------------------------------------------

with tab_eval:
    st.subheader("📊 Evaluation Benchmark")
    st.markdown(
        "Build and run the evaluation benchmark on open Romanian datasets "
        "(MLSUM + XLSum). Results are reported per contradiction type and "
        "separately for injected vs. real articles."
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    n_injected = col_e1.number_input("Injected samples", min_value=5, max_value=500, value=20)
    n_real = col_e2.number_input("Real samples", min_value=5, max_value=200, value=10)
    eval_seed = col_e3.number_input("Random seed", min_value=0, value=42)

    eval_btn = st.button("▶️ Run Evaluation", type="secondary")

    if eval_btn:
        with st.spinner("Building benchmark and evaluating… (this may take several minutes)"):
            try:
                from src.pipeline.evaluator import BenchmarkBuilder, Evaluator

                builder = BenchmarkBuilder(
                    n_injected=int(n_injected),
                    n_real=int(n_real),
                    seed=int(eval_seed),
                )
                samples = builder.build()

                if not samples:
                    st.error("No samples could be built — check dataset access.")
                    st.stop()

                pipeline_eval = RomanianPipeline(
                    backend=backend_ro,
                    use_wordnet=use_wordnet,
                    use_nli=use_nli,
                    nli_threshold=nli_threshold,
                )
                evaluator = Evaluator(pipeline=pipeline_eval)
                report = evaluator.run(samples)
                builder.save()
                evaluator.save_report(report)

            except Exception as exc:
                st.error(f"Evaluation error: {exc}")
                st.stop()

        ov = report["overall"]
        st.success(
            f"✅ Evaluation complete on {ov['total_samples']} samples — "
            f"F1={ov['f1']:.3f}  P={ov['precision']:.3f}  R={ov['recall']:.3f}"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{ov['precision']:.3f}")
        m2.metric("Recall", f"{ov['recall']:.3f}")
        m3.metric("F1", f"{ov['f1']:.3f}")
        m4.metric("TP / FP / FN", f"{ov['tp']} / {ov['fp']} / {ov['fn']}")

        col_inj, col_real = st.columns(2)
        with col_inj:
            inj = report["injected"]
            st.markdown("**Injected contradictions**")
            st.metric("F1", f"{inj['f1']:.3f}")
        with col_real:
            real = report["real"]
            st.markdown("**Real articles (no contradiction)**")
            st.metric("Precision", f"{real['precision']:.3f}")

        if report.get("by_type"):
            st.subheader("By contradiction type")
            import pandas as pd
            rows = [
                {"Type": t, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"]}
                for t, v in report["by_type"].items()
            ]
            st.dataframe(pd.DataFrame(rows).set_index("Type"))

        with st.expander("Raw report JSON"):
            st.code(json.dumps(report, indent=2, ensure_ascii=False), language="json")

        st.download_button(
            "Download full report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="eval_report.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Tab 4: About
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown("""
## About this project

**Internal Consistency of News Articles: Claim Extraction and Contradiction Detection for Romanian and Spanish**

**Team:** Bejinaru-Manoila Angel · Bocancea Georgiana-Letitia · Cojocaru Paul-Cristian · Herce Claudia

### Romanian Pipeline Architecture

| Phase | Component | Technology |
|---|---|---|
| Preprocessing | Tokenisation, POS, NER, Dep-parse | Teprolin + spaCy + Stanza |
| Claim Extraction | SPO triplets + numeric/temporal/NE attrs | Dependency parse heuristics |
| Normalisation | ISO 8601 dates, canonical units | Custom rule-based normaliser |
| Detection — Layer 1 | Numeric & temporal conflicts | Deterministic rules |
| Detection — Layer 1b | Antonymy & negated synonyms | Romanian WordNet (RoWordNet) |
| Detection — Layer 2 | Soft linguistic contradictions | XLM-RoBERTa (NLI) |
| Evaluation | P/R/F1 per type, injected vs real | MLSUM + XLSum benchmarks |
| Interface | Article submission + highlighting | Streamlit |

### Spanish Pipeline Architecture

| Phase | Component | Technology |
|---|---|---|
| Preprocessing | Tokenisation, POS, NER, Dep-parse | spaCy `es_core_news_lg` + Stanza |
| Claim Extraction | SPO triplets + numeric/temporal/NE attrs | Dependency parse heuristics |
| Normalisation | ISO 8601 dates, canonical units | Custom rule-based normaliser |
| Detection — Layer 1 | Numeric & temporal conflicts | Deterministic rules |
| Detection — Layer 1b | Antonymy & negated synonyms | NLTK Open Multilingual Wordnet (spa) |
| Detection — Layer 2 | Soft linguistic contradictions | XLM-RoBERTa (NLI) |
| Interface | Article submission + highlighting | Streamlit |

### Datasets (open, no registration)

| Dataset | Source | Role |
|---|---|---|
| MLSUM Romanian | `mlsum` / `ro` | Working set (Digi24 news articles) |
| XLSum Romanian | `csebuetnlp/xlsum` / `romanian` | Evaluation benchmark (BBC RO) |
| Romanian Wikipedia | `wikipedia` / `20231101.ro` | Factual article corpus |
| XNLI Romanian | `xnli` / `ro` | Contradiction label reference |
| FEVER | `fever` / `v1.0` | NLI methodology reference |

### Key resources

- **Romanian BERT**: `dumitrescuemilandrei/bert-base-romanian-cased-v1`
- **NLI model**: `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`
- **Romanian WordNet**: RoWordNet via `rowordnet` Python library
- **Spanish WordNet**: NLTK Open Multilingual Wordnet (`omw-1.4`, lang=`spa`)
    """)
