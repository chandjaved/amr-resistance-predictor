import io
import os
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import xgboost as xgb
import sys
from pathlib import Path

_AGENT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent / "code" / "app",
]

for _agent_dir in _AGENT_DIRS:
    if (_agent_dir / "Literature_Search_Agent.py").exists():
        sys.path.insert(0, str(_agent_dir))
        break

try:
    from Literature_Search_Agent import run_literature_research, paper_url
    LITERATURE_AGENT_IMPORT_ERROR = None
except Exception as e:
    run_literature_research = None
    paper_url = None
    LITERATURE_AGENT_IMPORT_ERROR = str(e)

MODEL_JSON = "models/prediction_model_corrected.json"
MODEL_PKL  = "models/prediction_model_corrected.pkl"
ENC_PKL    = "models/feature_encoders.pkl"


def _load_model():
    if os.path.exists(MODEL_JSON):
        try:
            m = xgb.XGBClassifier()
            m.load_model(MODEL_JSON)
            return m, None
        except Exception as e:
            return None, f"Failed to load JSON model: {e}"

    if os.path.exists(MODEL_PKL):
        try:
            return joblib.load(MODEL_PKL), None
        except Exception as e:
            return None, (
                "The .pkl model is incompatible with your current XGBoost "
                "version. Retrain with the updated train_model.py which saves "
                f"as .json. Details: {e}"
            )

    return None, f"No model found at {MODEL_JSON} or {MODEL_PKL}."


st.set_page_config(
    page_title="AMR Predictor — Research Edition",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AMR Surveillance Risk Predictor — research prototype.",
    },
)

try:
    st.set_option("client.showErrorDetails", False)
    st.set_option("client.toolbarMode", "minimal")
except Exception:
    pass


model, MODEL_LOAD_ERROR = _load_model()

if model is None:
    st.error("⚠️ The prediction model could not be loaded.")
    st.markdown(
        "**How to fix this:**\n\n"
        "1. Open a terminal in the project root.\n"
        "2. Run: `python code\\training\\train_model.py`\n"
        "3. This writes `models/prediction_model_corrected.json`.\n"
        "4. Refresh this page.\n\n"
        f"**Detail:** `{MODEL_LOAD_ERROR}`"
    )
    st.stop()

encoders = joblib.load(ENC_PKL)


def _safe_encode(le, value):
    return value if value in set(le.classes_) else le.classes_[0]


def prepare_features(country, specimen, pathogen, antibiotic):
    input_dict = {
        "CountryTerritoryArea": country,
        "Specimen":             specimen,
        "PathogenName":         pathogen,
        "AbTargets":            antibiotic,
    }
    input_df = pd.DataFrame([input_dict])

    for col in ["CountryTerritoryArea", "Specimen", "PathogenName", "AbTargets"]:
        input_df[col] = input_df[col].apply(lambda v: _safe_encode(encoders[col], v))
        input_df[col] = encoders[col].transform(input_df[col])

    return input_df[["CountryTerritoryArea", "Specimen", "PathogenName", "AbTargets"]]


COUNTRIES   = sorted(encoders["CountryTerritoryArea"].classes_.tolist())
PATHOGENS   = sorted(encoders["PathogenName"].classes_.tolist())
ANTIBIOTICS = sorted(encoders["AbTargets"].classes_.tolist())
SPECIMENS   = sorted(encoders["Specimen"].classes_.tolist())


st.markdown("""
<style>
:root {
    --clr-bg-app: #0F172A;
    --clr-bg-surface: #1E293B;
    --clr-text-main: #F1F5F9;
    --clr-text-muted: #94A3B8;
    --clr-brand-slate: #E2E8F0;
    --clr-brand-sky: #38BDF8;
    --clr-border: #334155;
}

#MainMenu                        { visibility: hidden !important; }
header[data-testid="stHeader"]   { display: none !important; }
footer                           { visibility: hidden !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stStatusWidget"]   { display: none !important; }
.stAppDeployButton               { display: none !important; }
[data-testid="stElementToolbar"] { display: none !important; }
button[title="View source code"],
button[aria-label="View source code"] { display: none !important; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--clr-bg-app) !important;
    color: var(--clr-text-main) !important;
}

[data-testid="stAppViewContainer"] { background-color: var(--clr-bg-app) !important; }
[data-testid="stHeader"]           { background: transparent !important; }

.block-container {
    max-width: 1280px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid var(--clr-border);
}

section[data-testid="stSidebar"] h2 {
    color: #F1F5F9;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] label {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background-color: var(--clr-brand-sky) !important;
    color: #0F172A !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background-color: #0EA5E9 !important;
    box-shadow: 0 4px 14px rgba(56,189,248,0.35) !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background-color: #334155 !important;
    color: #F1F5F9 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background-color: #475569 !important;
}

.app-header {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
    padding: 2rem 2.25rem !important;
    border-radius: 12px;
    margin-bottom: 1.25rem !important;
    border-left: 6px solid #38BDF8;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3),
                0 2px 4px -1px rgba(0,0,0,0.2);
}
.app-header h1 {
    color: #FFFFFF !important;
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.app-header p {
    color: #94A3B8 !important;
    margin-top: 0.5rem;
    margin-bottom: 0;
    font-size: 0.98rem;
}

.badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
.badge {
    background: #1E293B;
    color: #F1F5F9 !important;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    border: 1px solid #334155;
    transition: all 0.25s ease;
}
.badge:hover {
    background: #334155;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

.card {
    background: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.card:hover {
    box-shadow: 0 10px 28px rgba(0,0,0,0.5);
    transform: translateY(-2px);
}

.card p, .card h1, .card h2, .card h3, .card h4,
.card span, .card div, .card li {
    color: #F1F5F9 !important;
}

.risk-chip {
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.88rem;
}
.risk-low      { background: #064E3B; color: #6EE7B7 !important; border: 2px solid #10B981; }
.risk-moderate { background: #78350F; color: #FCD34D !important; border: 2px solid #F59E0B; }
.risk-high     { background: #7F1D1D; color: #FCA5A5 !important; border: 2px solid #EF4444; }

.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid #334155;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8 !important;
    font-weight: 500;
    padding: 0.55rem 0.9rem;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #38BDF8 !important;
    background: #1E293B;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    font-weight: 600;
}

h1, h2, h3, h4, p, span, li, label {
    color: #F1F5F9;
}

div[data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    transition: all 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: #F1F5F9 !important;
}

.stButton button {
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
}

.stDownloadButton button { border-radius: 8px !important; }

.streamlit-expanderHeader { color: #F1F5F9 !important; }
.streamlit-expanderHeader:hover { color: #38BDF8 !important; }

div[data-testid="stDataFrame"] { background: #1E293B !important; color: #F1F5F9 !important; }
div[data-testid="stDataFrame"] td:hover { background: #334155 !important; }

.stAlert {
    color: #F1F5F9 !important;
    background: #1E293B !important;
    border: 1px solid #334155 !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #94A3B8 !important;
}
</style>
""", unsafe_allow_html=True)


def render_gauge(prob_percent: float, color_hex: str, height: int = 160):
    radius = 70
    circumference = np.pi * radius
    offset = circumference * (1 - prob_percent / 100)

    svg_html = f"""
    <div style="text-align:center;margin:0;padding:0;width:100%;">
      <svg width="200" height="130" viewBox="0 0 180 110"
           style="max-width:100%;display:block;margin:0 auto;">
        <path d="M 20 90 A {radius} {radius} 0 0 1 160 90"
              fill="none" stroke="#334155" stroke-width="14"
              stroke-linecap="round"/>
        <path d="M 20 90 A {radius} {radius} 0 0 1 160 90"
              fill="none" stroke="{color_hex}" stroke-width="14"
              stroke-linecap="round"
              stroke-dasharray="{circumference}"
              stroke-dashoffset="{offset}">
          <animate attributeName="stroke-dashoffset"
                   from="{circumference}" to="{offset}"
                   dur="1s" fill="freeze"/>
        </path>
        <text x="90" y="80" text-anchor="middle"
              font-family="'IBM Plex Mono', monospace"
              font-size="26" font-weight="600" fill="{color_hex}">
            {prob_percent:.1f}%
        </text>
        <text x="90" y="100" text-anchor="middle"
              font-family="Inter, sans-serif"
              font-size="11" fill="#94A3B8" font-weight="500">
            resistance probability
        </text>
      </svg>
    </div>
    """
    components.html(svg_html, height=height)


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


metrics         = _load_json("results/metrics/metrics_corrected.json")
model_card      = _load_json("results/metrics/model_card.json")
shap_importance = _load_json("results/metrics/shap_global_importance.json")


if "history" not in st.session_state:
    st.session_state.history = []


def risk_category(prob_percent: float):
    if prob_percent < 30:
        return "Low Risk", "green"
    elif prob_percent < 60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"


def clinical_interpretation(country, pathogen, antibiotic, prob_percent, alt_probs):
    label, _ = risk_category(prob_percent)
    best_alt = min(alt_probs, key=lambda x: x[1])

    text = (
        f"The model estimates a **{prob_percent:.1f}%** probability that "
        f"reported **{pathogen}** resistance to **{antibiotic}** in "
        f"**{country}** falls into the High Risk surveillance category "
        f"({label.lower()}; High Risk = reported resistance ≥ 20%, "
        f"an investigator-defined threshold). "
    )
    if best_alt[0] != antibiotic and best_alt[1] < prob_percent - 5:
        text += (
            f"Among the antibiotics considered, **{best_alt[0]}** "
            f"shows the lowest predicted high-risk probability "
            f"({best_alt[1]:.1f}%) for this pathogen and country, "
            "which may be relevant to surveillance or stewardship "
            "prioritization — this is a surveillance-level signal, "
            "not a treatment recommendation for any individual patient."
        )
    else:
        text += ("No clearly lower-risk alternative was identified among "
                 "the antibiotics considered.")
    return text


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


def local_shap_contributions(features_df):
    try:
        explainer = get_explainer(model)
        sv = explainer.shap_values(features_df)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.array(sv).reshape(-1)
        return pd.DataFrame({
            "Feature": features_df.columns,
            "Value": features_df.iloc[0].values,
            "SHAP contribution": sv,
        }).sort_values("SHAP contribution", key=abs, ascending=False)
    except Exception:
        return None


def build_pdf_report(patient, prob_percent, label, interpretation) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    import textwrap

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "AMR Resistance Prediction Report (Research Prototype)")
    y -= 1 * cm
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, y,
                 f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 0.8 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Input Summary")
    y -= 0.6 * cm
    c.setFont("Helvetica", 10)
    for k, v in patient.items():
        c.drawString(2.2 * cm, y, f"{k}: {v}")
        y -= 0.5 * cm

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Prediction")
    y -= 0.6 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2.2 * cm, y,
                 f"Predicted resistance probability: {prob_percent:.1f}%  ({label})")
    y -= 0.8 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Clinical Interpretation")
    y -= 0.6 * cm
    c.setFont("Helvetica", 9)
    for line in textwrap.wrap(interpretation.replace("**", ""), 95):
        c.drawString(2.2 * cm, y, line)
        y -= 0.45 * cm

    y -= 0.8 * cm
    c.setFont("Helvetica-Oblique", 8)
    for line in [
        "DISCLAIMER: This output is generated by a machine learning prototype trained on",
        "synthetic data for educational and research purposes only. It is NOT a substitute",
        "for clinical judgement, culture and susceptibility testing, or institutional",
        "antimicrobial stewardship guidance.",
    ]:
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


st.markdown("""
<div class="app-header">
  <h1>🦠 AMR Surveillance Risk Predictor</h1>
  <p>Country-level antimicrobial resistance risk prediction
     from WHO GLASS 2022 surveillance data ·
     Educational &amp; research use only</p>
</div>
""", unsafe_allow_html=True)


_version = "2.0-real-data"
_n = "6,402"

st.markdown(f"""
<div class="badge-row">
  <span class="badge">🧠 SHAP Explainable AI</span>
  <span class="badge">📄 Model Card v{_version}</span>
  <span class="badge">🌍 WHO GLASS 2022 Real Data (n={_n})</span>
  <span class="badge">📊 Performance Telemetry Active</span>
  <span class="badge">⚕️ Research Blueprint</span>
</div>
""", unsafe_allow_html=True)


with st.expander("ℹ️ About this tool, its data, and its limitations", expanded=False):
    st.markdown(
        "This tool predicts a **country-level surveillance risk category** "
        "(\"High Risk\" = reported resistance ≥ 20%, an investigator-defined "
        "threshold, not an official WHO cutoff) from country, specimen type, "
        "pathogen, and antibiotic, trained on real 2017–2020 data compiled "
        "from the WHO GLASS 2022 report (WHO, 2022; CC BY-NC-SA 3.0 IGO). "
        "**Each row is an aggregate surveillance report, not an individual patient.** "
        "**Not for clinical use.**"
    )


with st.sidebar:
    st.header("Surveillance Query")

    if st.button("🎲 Load Random Sample Query", use_container_width=True):
        rng = np.random.default_rng()
        st.session_state["_sample"] = {
            "country":    rng.choice(COUNTRIES),
            "antibiotic": rng.choice(ANTIBIOTICS),
            "specimen":   rng.choice(SPECIMENS),
            "pathogen":   rng.choice(PATHOGENS),
        }

    sample = st.session_state.get("_sample", {})

    country = st.selectbox(
        "Country / Territory", COUNTRIES,
        index=(COUNTRIES.index(sample["country"]) if sample.get("country") in COUNTRIES else 0)
    )
    pathogen = st.selectbox(
        "Pathogen", PATHOGENS,
        index=(PATHOGENS.index(sample["pathogen"]) if sample.get("pathogen") in PATHOGENS else 0)
    )
    antibiotic = st.selectbox(
        "Antibiotic", ANTIBIOTICS,
        index=(ANTIBIOTICS.index(sample["antibiotic"]) if sample.get("antibiotic") in ANTIBIOTICS else 0)
    )
    specimen = st.selectbox(
        "Specimen type", SPECIMENS,
        index=(SPECIMENS.index(sample["specimen"]) if sample.get("specimen") in SPECIMENS else 0)
    )

    predict_clicked = st.button(
        "Predict Resistance Risk", type="primary", use_container_width=True
    )

    st.markdown("---")
    st.caption(
        "Predicts a country-level surveillance category, not an individual "
        "patient outcome. Educational/research prototype only. Not for clinical use."
    )


(tab_predict, tab_perf, tab_shap, tab_batch, tab_history, tab_about, tab_literature) = st.tabs([
    "🔬 Predict",
    "📊 Model Performance",
    "🧠 Explainability",
    "📁 Batch Prediction",
    "🕒 History",
    "📖 About & Model Card",
    "📚 Literature Search Agent",
])


with tab_predict:

    if predict_clicked:
        features = prepare_features(country, specimen, pathogen, antibiotic)
        proba = model.predict_proba(features)[0]
        prob_percent = float(proba[1] * 100)
        label, color = risk_category(prob_percent)

        color_hex  = {"green": "#10B981", "orange": "#F59E0B", "red": "#EF4444"}[color]
        risk_class = {"green": "risk-low", "orange": "risk-moderate", "red": "risk-high"}[color]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Result")
            render_gauge(prob_percent, color_hex)
            st.markdown(
                f'<div style="text-align:center;margin-top:0.4rem;">'
                f'<span class="risk-chip {risk_class}">{label}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Resistance Context Matrix")

            probs = []
            for ab in ANTIBIOTICS:
                feat = prepare_features(country, specimen, pathogen, ab)
                p = float(model.predict_proba(feat)[0][1] * 100)
                probs.append((ab, p))

            chart_data = pd.DataFrame(
                probs, columns=["Antibiotic", "High-Risk Probability (%)"]
            ).set_index("Antibiotic")

            st.bar_chart(chart_data, color="#38BDF8")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🩺 Surveillance Interpretation Summary")
        interpretation = clinical_interpretation(
            country, pathogen, antibiotic, prob_percent, probs
        )
        st.info(interpretation)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Local SHAP Contributions Framework")
        contrib = local_shap_contributions(features)
        if contrib is not None:
            st.dataframe(
                contrib.style.format({"SHAP contribution": "{:+.3f}"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("SHAP explanation mapping unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "country": country,
            "specimen": specimen,
            "pathogen": pathogen,
            "antibiotic": antibiotic,
            "high_risk_probability_%": round(prob_percent, 1),
            "risk_category": label,
        })

        st.subheader("📥 Export Pipeline Data")
        query = {"Country": country, "Specimen": specimen,
                 "Pathogen": pathogen, "Antibiotic": antibiotic}

        csv_bytes = pd.DataFrame([{
            **query,
            "High_Risk_Probability_%": round(prob_percent, 1),
            "Risk_Category": label,
        }]).to_csv(index=False).encode()

        colA, colB = st.columns(2)
        with colA:
            st.download_button("⬇️ Download as CSV Matrix", csv_bytes,
                               file_name="amr_prediction.csv", mime="text/csv")
        with colB:
            try:
                pdf_bytes = build_pdf_report(query, prob_percent, label, interpretation)
                st.download_button("⬇️ Download PDF Report Artifact", pdf_bytes,
                                   file_name="amr_prediction_report.pdf",
                                   mime="application/pdf")
            except Exception as e:
                st.caption(f"PDF engine failed: {e}")

    else:
        st.info(
            "Set the country / specimen / pathogen / antibiotic query "
            "in the sidebar and click **Predict Resistance Risk** to begin."
        )


with tab_perf:
    st.subheader("Held-out Test Performance Configuration")

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy Matrix", f"{metrics['accuracy'] * 100:.1f}%")
        c2.metric("ROC-AUC Score",   f"{metrics['roc_auc']:.3f}")
        c3.metric("PR-AUC Score",    f"{metrics['pr_auc']:.3f}")
        c4.metric("F1-Score Metrics", f"{metrics['f1_score']:.3f}")
        if "brier_score" in metrics:
            c5.metric("Brier Score", f"{metrics['brier_score']:.3f}")

        st.markdown("---")
        colA, colB, colC = st.columns(3)
        for col, img, cap in [
            (colA, "results/figures/roc_curve_corrected.png", "ROC Curve Map"),
            (colB, "results/figures/pr_curve.png", "Precision–Recall Curve Analysis"),
            (colC, "results/figures/confusion_matrix.png", "Confusion Matrix Breakdown"),
        ]:
            if os.path.exists(img):
                col.image(img, caption=cap, use_container_width=True)
            else:
                col.caption(f"{cap} asset missing from disk volume paths.")
    else:
        st.warning("Telemetry descriptor structure metrics.json not located.")


with tab_shap:
    st.subheader("Global Explainer Feature Profiles (SHAP Engine)")

    colA, colB = st.columns(2)
    for col, img, cap in [
        (colA, "results/figures/shap_bar.png",     "Mean |SHAP value| Performance Graph"),
        (colB, "results/figures/shap_summary.png", "SHAP Positional Summary Distribution Diagram"),
    ]:
        if os.path.exists(img):
            col.image(img, caption=cap, use_container_width=True)
        else:
            col.caption(f"{cap} asset missing from disk path storage.")

    fi_path = "results/figures/feature_importance.png"
    if os.path.exists(fi_path):
        st.image(fi_path, caption="XGBoost Feature Importance", use_container_width=True)


with tab_batch:
    st.subheader("Cohort Optimization Bulk Operations Block")

    st.download_button(
        "⬇️ Download Core CSV Template",
        pd.DataFrame([{
            "country": "Kenya", "specimen": "URINE",
            "pathogen": "Escherichia coli", "antibiotic": "Ciprofloxacin",
        }]).to_csv(index=False).encode(),
        file_name="batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload Target Batch Processing Pipeline Records (CSV)", type=["csv"]
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            results = []
            for _, row in batch_df.iterrows():
                feat = prepare_features(
                    row["country"], row["specimen"],
                    row["pathogen"], row["antibiotic"],
                )
                p = float(model.predict_proba(feat)[0][1] * 100)
                label, _ = risk_category(p)
                results.append({
                    **row.to_dict(),
                    "resistance_probability_%": round(p, 1),
                    "risk_category": label,
                })
            st.dataframe(pd.DataFrame(results),
                         use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Failed parsing uploaded batch context layers: {e}")


with tab_history:
    st.subheader("Session Tracking Logs")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No query tracking sequences logged within this context instance loop yet.")


with tab_about:
    st.subheader("Model Card Specification Ledger")
    if model_card:
        st.json(model_card)
    else:
        st.markdown("No complete model card architecture profile template available on system paths.")
with tab_literature:
    st.subheader("📚 AMR Literature Search Agent")

    st.write(
        "Search scientific literature related to antimicrobial resistance "
        "using Europe PMC. The agent retrieves research papers, identifies "
        "potential evidence gaps, and supports iterative literature searching."
    )

    if LITERATURE_AGENT_IMPORT_ERROR:
        st.error("The Literature Search Agent could not be imported.")
        st.code(LITERATURE_AGENT_IMPORT_ERROR)

    else:
        literature_question = st.text_area(
            "Enter your research question",
            value="What mechanisms contribute to antimicrobial resistance in Escherichia coli?",
            height=110,
            key="literature_question",
        )

        col1, col2 = st.columns(2)

        with col1:
            max_iterations = st.number_input(
                "Maximum search iterations",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="literature_iterations",
            )

        with col2:
            results_per_search = st.number_input(
                "Results per search",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="literature_results",
            )

        search_clicked = st.button(
            "🔎 Start Literature Search",
            type="primary",
            use_container_width=True,
            key="start_literature_search",
        )

        if search_clicked:
            if not literature_question.strip():
                st.warning("Please enter a research question.")

            else:
                try:
                    with st.spinner(
                        "Searching Europe PMC and analyzing retrieved literature..."
                    ):
                        literature_result = run_literature_research(
                            question=literature_question.strip(),
                            max_iterations=int(max_iterations),
                            results_per_search=int(results_per_search),
                        )

                    st.session_state["literature_result"] = literature_result

                except Exception as e:
                    st.error(f"Literature search failed: {e}")

        literature_result = st.session_state.get("literature_result")

        if literature_result is not None:

            if isinstance(literature_result, dict):
                result = literature_result
            else:
                result = vars(literature_result)

            search_queries = result.get("search_queries", [])
            papers = result.get("papers", [])
            gaps = result.get("gaps", [])
            errors = result.get("errors", [])
            status = result.get("status", "Completed")
            iteration = result.get("iteration", 0)

            st.markdown("---")
            st.subheader("Research Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Papers Retrieved", len(papers))
            col2.metric("Search Iterations", iteration)
            col3.metric("Potential Evidence Gaps", len(gaps))

            st.write("**Agent Status:**", status)

            if search_queries:
                with st.expander("Search Queries", expanded=True):
                    for i, query in enumerate(search_queries, start=1):
                        st.write(f"{i}. {query}")

            st.markdown("### Retrieved Research Papers")

            if papers:
                for i, paper in enumerate(papers, start=1):

                    if isinstance(paper, dict):
                        title = paper.get("title", "Title unavailable")
                        year = paper.get("year", "Year unavailable")
                        doi = paper.get("doi", "")
                        pmid = paper.get("pmid", "")
                        abstract = paper.get("abstract", "")
                    else:
                        title = getattr(paper, "title", "Title unavailable")
                        year = getattr(paper, "year", "Year unavailable")
                        doi = getattr(paper, "doi", "")
                        pmid = getattr(paper, "pmid", "")
                        abstract = getattr(paper, "abstract", "")

                    with st.expander(f"{i}. {title}"):

                        st.write(f"**Publication year:** {year}")

                        if doi:
                            st.write(f"**DOI:** {doi}")

                        if pmid:
                            st.write(f"**PMID:** {pmid}")

                        if abstract:
                            st.write("**Abstract:**")
                            st.write(abstract)
                        else:
                            st.caption("Abstract unavailable.")

                        if paper_url is not None:
                            try:
                                url = paper_url(paper)
                                if url:
                                    st.markdown(
                                        f"[Open research paper]({url})"
                                    )
                            except Exception:
                                pass

            else:
                st.info("No research papers were retrieved.")

            st.markdown("### Potential Evidence Gaps")

            if gaps:
                for gap in gaps:
                    st.write(f"• {gap}")
            else:
                st.info("No potential evidence gaps were identified.")

            if errors:
                with st.expander("Search Errors"):
                    for error in errors:
                        st.warning(str(error))

            st.caption(
                "Evidence gaps are candidate areas not sufficiently represented "
                "in the retrieved results. They do not establish that evidence "
                "is absent from the wider scientific literature."
            )
