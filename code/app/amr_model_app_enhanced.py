
import io
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# LOAD MODEL AND ENCODERS
# ============================================================

model = joblib.load("models/prediction_model_corrected.pkl")
encoders = joblib.load("models/feature_encoders.pkl")


# ============================================================
# SAFE ENCODING
# ============================================================

def _safe_encode(le, value):
    """Map unseen categories to the encoder's first known class
    rather than crash, since a random train/test split can leave
    rare country/pathogen/antibiotic combinations out of the
    training fold.
    """
    return value if value in set(le.classes_) else le.classes_[0]


def prepare_features(country, specimen, pathogen, antibiotic):
    input_dict = {
        "CountryTerritoryArea": country,
        "Specimen": specimen,
        "PathogenName": pathogen,
        "AbTargets": antibiotic,
    }

    input_df = pd.DataFrame([input_dict])

    for col in [
        "CountryTerritoryArea",
        "Specimen",
        "PathogenName",
        "AbTargets",
    ]:
        input_df[col] = input_df[col].apply(
            lambda v: _safe_encode(encoders[col], v)
        )
        input_df[col] = encoders[col].transform(input_df[col])

    input_df = input_df[
        [
            "CountryTerritoryArea",
            "Specimen",
            "PathogenName",
            "AbTargets",
        ]
    ]

    return input_df


# ============================================================
# LOAD ENCODER CATEGORIES
# ============================================================

COUNTRIES = sorted(
    encoders["CountryTerritoryArea"].classes_.tolist()
)

PATHOGENS = sorted(
    encoders["PathogenName"].classes_.tolist()
)

ANTIBIOTICS = sorted(
    encoders["AbTargets"].classes_.tolist()
)

SPECIMENS = sorted(
    encoders["Specimen"].classes_.tolist()
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AMR Predictor — Research Edition",
    page_icon="🦠",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

:root {
    --clr-bg-app: #F8FAFC;
    --clr-bg-surface: #FFFFFF;
    --clr-text-main: #0F172A;
    --clr-text-muted: #475569;
    --clr-text-light: #FFFFFF;
    --clr-brand-slate: #1E293B;
    --clr-brand-sky: #0EA5E9;
    --clr-border: #E2E8F0;
    --clr-status-success: #10B981;
    --clr-status-warning: #F59E0B;
    --clr-status-error: #EF4444;
}

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
}

section[data-testid="stSidebar"] label {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #0F172A !important;
}

.app-header {
    background: linear-gradient(
        135deg,
        #1E293B 0%,
        #334155 100%
    ) !important;

    padding: 2.25rem !important;
    border-radius: 12px;
    margin-bottom: 1.5rem !important;
    border-left: 6px solid #0EA5E9;

    box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.1),
        0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.app-header h1 {
    color: #FFFFFF !important;
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}

.app-header p {
    color: #CBD5E1 !important;
    margin-top: 0.5rem;
    font-size: 1rem;
}

.app-header h1:hover {
    transform: scale(1.02);
    transition: transform 0.2s ease;
}

.app-header p:hover {
    color: #E2E8F0 !important;
    transition: color 0.2s ease;
}

.badge-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}

.badge {
    background: #F1F5F9;
    color: #1E293B !important;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid #E2E8F0;
    transition: all 0.3s ease;
}

.badge:hover {
    background: #E2E8F0;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    cursor: pointer;
}

.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.risk-chip {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}

.risk-low {
    background: #D1FAE5;
    color: #065F46 !important;
    border: 2px solid #10B981;
}

.risk-moderate {
    background: #FEF3C7;
    color: #92400E !important;
    border: 2px solid #F59E0B;
}

.risk-high {
    background: #FEE2E2;
    color: #991B1B !important;
    border: 2px solid #EF4444;
}

.risk-chip:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #0EA5E9 !important;
    background: #F1F5F9;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #0EA5E9 !important;
    font-weight: 600;
}

.stSubheader:hover {
    color: #0EA5E9 !important;
    transition: color 0.2s ease;
}

div[data-testid="stMetric"] {
    transition: all 0.3s ease;
    border-radius: 8px;
    padding: 0.5rem;
}

div[data-testid="stMetric"]:hover {
    background: #F8FAFC;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.stButton button {
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

.streamlit-expanderHeader:hover {
    color: #0EA5E9 !important;
}

div[data-testid="stDataFrame"] td:hover {
    background: #F1F5F9 !important;
    transition: background 0.2s ease;
}

div[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: #0EA5E9 !important;
    color: white !important;
}

.stAlert:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: box-shadow 0.2s ease;
}

p:hover,
span:hover,
li:hover {
    transition: opacity 0.2s ease;
}

.stMarkdown {
    color: inherit;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GAUGE
# ============================================================

def gauge_svg(prob_percent: float, color_hex: str) -> str:

    radius = 70
    circumference = np.pi * radius
    offset = circumference * (1 - prob_percent / 100)

    return f"""
    <div style="
        text-align:center;
        margin:0;
        padding:0;
        width:100%;
        transition: all 0.3s ease;"
        onmouseover="this.style.transform='scale(1.05)'"
        onmouseout="this.style.transform='scale(1)'">

      <svg width="180"
           height="110"
           viewBox="0 0 180 110"
           style="
               max-width:100%;
               display:block;
               margin:0 auto;">

        <path
            d="M 20 90 A {radius} {radius} 0 0 1 160 90"
            fill="none"
            stroke="#E2E8F0"
            stroke-width="14"
            stroke-linecap="round"/>

        <path
            d="M 20 90 A {radius} {radius} 0 0 1 160 90"
            fill="none"
            stroke="{color_hex}"
            stroke-width="14"
            stroke-linecap="round"
            stroke-dasharray="{circumference}"
            stroke-dashoffset="{offset}">

          <animate
              attributeName="stroke-dashoffset"
              from="{circumference}"
              to="{offset}"
              dur="1s"
              fill="freeze"/>

        </path>

        <text
            x="90"
            y="80"
            text-anchor="middle"
            font-family="IBM Plex Mono, monospace"
            font-size="26"
            font-weight="600"
            fill="{color_hex}">
            {prob_percent:.1f}%
        </text>

        <text
            x="90"
            y="100"
            text-anchor="middle"
            font-family="Inter, sans-serif"
            font-size="11"
            fill="#64748B"
            font-weight="500">
            resistance probability
        </text>

      </svg>
    </div>
    """


# ============================================================
# LOAD JSON FILES
# ============================================================

def _load_json(path):

    try:
        with open(path) as f:
            return json.load(f)

    except Exception:
        return None


metrics = _load_json(
    "results/metrics/metrics_corrected.json"
)

model_card = _load_json(
    "results/metrics/model_card.json"
)

shap_importance = _load_json(
    "results/metrics/shap_global_importance.json"
)


# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(prob_percent: float):

    if prob_percent < 30:
        return "Low Risk", "green"

    elif prob_percent < 60:
        return "Moderate Risk", "orange"

    else:
        return "High Risk", "red"


# ============================================================
# CLINICAL INTERPRETATION
# ============================================================

def clinical_interpretation(
    country,
    pathogen,
    antibiotic,
    prob_percent,
    alt_probs
):

    label, _ = risk_category(prob_percent)

    best_alt = min(
        alt_probs,
        key=lambda x: x[1]
    )

    text = (
        f"The model estimates a **{prob_percent:.1f}%** probability that "
        f"reported **{pathogen}** resistance to **{antibiotic}** in "
        f"**{country}** falls into the High Risk surveillance category "
        f"({label.lower()}; High Risk = reported resistance ≥ 20%, "
        f"an investigator-defined threshold). "
    )

    if (
        best_alt[0] != antibiotic
        and best_alt[1] < prob_percent - 5
    ):

        text += (
            f"Among the antibiotics considered, **{best_alt[0]}** "
            f"shows the lowest predicted high-risk probability "
            f"({best_alt[1]:.1f}%) for this pathogen and country, "
            "which may be relevant to surveillance or stewardship "
            "prioritization — this is a surveillance-level signal, "
            "not a treatment recommendation for any individual patient."
        )

    else:

        text += (
            "No clearly lower-risk alternative was identified among "
            "the antibiotics considered."
        )

    return text


# ============================================================
# SHAP EXPLAINER
# ============================================================

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

        out = pd.DataFrame({
            "Feature": features_df.columns,
            "Value": features_df.iloc[0].values,
            "SHAP contribution": sv,
        }).sort_values(
            "SHAP contribution",
            key=abs,
            ascending=False
        )

        return out

    except Exception:

        return None


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf_report(
    patient,
    prob_percent,
    label,
    interpretation
) -> bytes:

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    import textwrap

    buf = io.BytesIO()

    c = canvas.Canvas(
        buf,
        pagesize=A4
    )

    width, height = A4

    y = height - 2 * cm

    c.setFont(
        "Helvetica-Bold",
        16
    )

    c.drawString(
        2 * cm,
        y,
        "AMR Resistance Prediction Report (Research Prototype)"
    )

    y -= 1 * cm

    c.setFont(
        "Helvetica",
        9
    )

    c.drawString(
        2 * cm,
        y,
        f"Generated: "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    y -= 0.8 * cm

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        2 * cm,
        y,
        "Input Summary"
    )

    y -= 0.6 * cm

    c.setFont(
        "Helvetica",
        10
    )

    for k, v in patient.items():

        c.drawString(
            2.2 * cm,
            y,
            f"{k}: {v}"
        )

        y -= 0.5 * cm

    y -= 0.3 * cm

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        2 * cm,
        y,
        "Prediction"
    )

    y -= 0.6 * cm

    c.setFont(
        "Helvetica",
        10
    )

    c.drawString(
        2.2 * cm,
        y,
        f"Predicted resistance probability: "
        f"{prob_percent:.1f}%  ({label})"
    )

    y -= 0.8 * cm

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        2 * cm,
        y,
        "Clinical Interpretation"
    )

    y -= 0.6 * cm

    c.setFont(
        "Helvetica",
        9
    )

    for line in textwrap.wrap(
        interpretation.replace("**", ""),
        95
    ):

        c.drawString(
            2.2 * cm,
            y,
            line
        )

        y -= 0.45 * cm

    y -= 0.8 * cm

    c.setFont(
        "Helvetica-Oblique",
        8
    )

    for line in [
        "DISCLAIMER: This output is generated by a machine learning prototype trained on",
        "synthetic data for educational and research purposes only. It is NOT a substitute",
        "for clinical judgement, culture and susceptibility testing, or institutional",
        "antimicrobial stewardship guidance.",
    ]:

        c.drawString(
            2 * cm,
            y,
            line
        )

        y -= 0.4 * cm

    c.showPage()

    c.save()

    buf.seek(0)

    return buf.read()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="app-header">

  <h1>🦠 AMR Surveillance Risk Predictor</h1>

  <p>
    Country-level antimicrobial resistance risk prediction
    from WHO GLASS 2022 surveillance data ·
    Educational &amp; research use only
  </p>

</div>
""", unsafe_allow_html=True)


_version = "2.0-real-data"
_n = "6,402"


st.markdown(f"""
<div class="badge-row">

  <span class="badge">
    🧠 SHAP Explainable AI
  </span>

  <span class="badge">
    📄 Model Card v{_version}
  </span>

  <span class="badge">
    🌍 WHO GLASS 2022 Real Data (n={_n})
  </span>

  <span class="badge">
    📊 Performance Telemetry Active
  </span>

  <span class="badge">
    ⚕️ Research Blueprint
  </span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# ABOUT TOOL
# ============================================================

with st.expander(
    "ℹ️ About this tool, its data, and its limitations",
    expanded=False
):

    st.markdown(
        "This tool predicts a **country-level surveillance risk category** "
        "(\"High Risk\" = reported resistance ≥ 20%, an investigator-defined "
        "threshold, not an official WHO cutoff) from country, specimen type, "
        "pathogen, and antibiotic, trained on real 2017–2020 data compiled "
        "from the WHO GLASS 2022 report (World Health Organization, 2022; "
        "CC BY-NC-SA 3.0 IGO). **Each row is an aggregate surveillance report, "
        "not an individual patient** — this tool does not predict any single "
        "patient's resistance status and must not be used for individual "
        "clinical decisions. Coverage is limited to countries with continuous "
        "2017–2020 reporting or 2019 data specifically; results are not a "
        "complete or fully representative global picture. **Not for clinical use.**"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Surveillance Query")

    if st.button(
        "🎲 Load Random Sample Query",
        use_container_width=True
    ):

        rng = np.random.default_rng()

        st.session_state["_sample"] = {

            "country": rng.choice(COUNTRIES),

            "antibiotic": rng.choice(ANTIBIOTICS),

            "specimen": rng.choice(SPECIMENS),

            "pathogen": rng.choice(PATHOGENS),

        }

    sample = st.session_state.get(
        "_sample",
        {}
    )

    country = st.selectbox(
        "Country / Territory",
        COUNTRIES,
        index=(
            COUNTRIES.index(sample["country"])
            if sample.get("country") in COUNTRIES
            else 0
        )
    )

    pathogen = st.selectbox(
        "Pathogen",
        PATHOGENS,
        index=(
            PATHOGENS.index(sample["pathogen"])
            if sample.get("pathogen") in PATHOGENS
            else 0
        )
    )

    antibiotic = st.selectbox(
        "Antibiotic",
        ANTIBIOTICS,
        index=(
            ANTIBIOTICS.index(sample["antibiotic"])
            if sample.get("antibiotic") in ANTIBIOTICS
            else 0
        )
    )

    specimen = st.selectbox(
        "Specimen type",
        SPECIMENS,
        index=(
            SPECIMENS.index(sample["specimen"])
            if sample.get("specimen") in SPECIMENS
            else 0
        )
    )

    predict_clicked = st.button(
        "Predict Resistance Risk",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")

    st.caption(
        "Predicts a country-level surveillance category, "
        "not an individual patient outcome. Educational/research "
        "prototype only. Not for clinical use. See "
        "'About & Model Card' tab for full disclaimers."
    )


# ============================================================
# TABS
# ============================================================

(
    tab_predict,
    tab_perf,
    tab_shap,
    tab_batch,
    tab_history,
    tab_about,
) = st.tabs([
    "🔬 Predict",
    "📊 Model Performance",
    "🧠 Explainability",
    "📁 Batch Prediction",
    "🕒 History",
    "📖 About & Model Card"
])


# ============================================================
# PREDICTION TAB
# ============================================================

with tab_predict:

    if predict_clicked:

        features = prepare_features(
            country,
            specimen,
            pathogen,
            antibiotic
        )

        proba = model.predict_proba(features)[0]

        resistant_prob = proba[1]

        prob_percent = resistant_prob * 100

        label, color = risk_category(
            prob_percent
        )

        color_hex = {
            "green": "#10B981",
            "orange": "#F59E0B",
            "red": "#EF4444"
        }[color]

        risk_class = {
            "green": "risk-low",
            "orange": "risk-moderate",
            "red": "risk-high"
        }[color]

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Prediction Result"
            )

            st.markdown(
                gauge_svg(
                    prob_percent,
                    color_hex
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="text-align:center;margin-top:0.6rem;">'
                f'<span class="risk-chip {risk_class}">'
                f'{label}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Resistance Context Matrix"
            )

            probs = []

            for ab in ANTIBIOTICS:

                feat = prepare_features(
                    country,
                    specimen,
                    pathogen,
                    ab
                )

                p = (
                    model.predict_proba(feat)[0][1]
                    * 100
                )

                probs.append(
                    (ab, p)
                )

            chart_data = pd.DataFrame(
                probs,
                columns=[
                    "Antibiotic",
                    "High-Risk Probability (%)"
                ]
            ).set_index(
                "Antibiotic"
            )

            st.bar_chart(
                chart_data,
                color="#0EA5E9"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🩺 Surveillance Interpretation Summary"
        )

        interpretation = clinical_interpretation(
            country,
            pathogen,
            antibiotic,
            prob_percent,
            probs
        )

        st.info(
            interpretation
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🧠 Local SHAP Contributions Framework"
        )

        contrib = local_shap_contributions(
            features
        )

        if contrib is not None:

            st.dataframe(
                contrib.style.format({
                    "SHAP contribution": "{:+.3f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.caption(
                "SHAP explanation mapping unavailable."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.session_state.history.append({

            "timestamp":
                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "country": country,

            "specimen": specimen,

            "pathogen": pathogen,

            "antibiotic": antibiotic,

            "high_risk_probability_%":
                round(prob_percent, 1),

            "risk_category": label,

        })

        st.subheader(
            "📥 Export Pipeline Data"
        )

        query = {

            "Country": country,

            "Specimen": specimen,

            "Pathogen": pathogen,

            "Antibiotic": antibiotic

        }

        csv_bytes = pd.DataFrame([{

            **query,

            "High_Risk_Probability_%":
                round(prob_percent, 1),

            "Risk_Category": label

        }]).to_csv(
            index=False
        ).encode()

        colA, colB = st.columns(2)

        with colA:

            st.download_button(
                "⬇️ Download as CSV Matrix",
                csv_bytes,
                file_name="amr_prediction.csv",
                mime="text/csv"
            )

        with colB:

            try:

                pdf_bytes = build_pdf_report(
                    query,
                    prob_percent,
                    label,
                    interpretation
                )

                st.download_button(
                    "⬇️ Download PDF Report Artifact",
                    pdf_bytes,
                    file_name="amr_prediction_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:

                st.caption(
                    f"PDF engine failed context initialization: {e}"
                )

    else:

        st.info(
            "Set the country/specimen/pathogen/antibiotic "
            "query in the sidebar and click "
            "**Predict Resistance Risk** to begin."
        )


# ============================================================
# PERFORMANCE TAB
# ============================================================

with tab_perf:

    st.subheader(
        "Held-out Test Performance Configuration"
    )

    if metrics:

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy Matrix",
            f"{metrics['accuracy'] * 100:.1f}%"
        )

        c2.metric(
            "ROC-AUC Score",
            f"{metrics['roc_auc']:.3f}"
        )

        c3.metric(
            "PR-AUC Score",
            f"{metrics['pr_auc']:.3f}"
        )

        c4.metric(
            "F1-Score Metrics",
            f"{metrics['f1_score']:.3f}"
        )

        if "brier_score" in metrics:

            c5.metric(
                "Brier Score",
                f"{metrics['brier_score']:.3f}"
            )

        st.markdown("---")

        colA, colB, colC = st.columns(3)

        for col, img, cap in [

            (
                colA,
                "results/figures/roc_curve_corrected.png",
                "ROC Curve Map"
            ),

            (
                colB,
                "results/figures/pr_curve.png",
                "Precision–Recall Curve Analysis"
            ),

            (
                colC,
                "results/figures/confusion_matrix.png",
                "Confusion Matrix Breakdown"
            ),

        ]:

            try:

                col.image(
                    img,
                    caption=cap,
                    use_container_width=True
                )

            except Exception:

                col.caption(
                    f"{cap} asset missing from disk volume paths."
                )

    else:

        st.warning(
            "Telemetry descriptor structure "
            "metrics.json not located."
        )


# ============================================================
# SHAP TAB
# ============================================================

with tab_shap:

    st.subheader(
        "Global Explainer Feature Profiles (SHAP Engine)"
    )

    colA, colB = st.columns(2)

    try:

        colA.image(
            "results/figures/shap_bar.png",
            caption="Mean |SHAP value| Performance Graph",
            use_container_width=True
        )

    except Exception:

        colA.caption(
            "shap_bar.png visualization target absent "
            "from disk path storage."
        )

    try:

        colB.image(
            "results/figures/shap_summary.png",
            caption="SHAP Positional Summary Distribution Diagram",
            use_container_width=True
        )

    except Exception:

        colB.caption(
            "shap_summary.png visualization target absent "
            "from disk path storage."
        )


# ============================================================
# BATCH PREDICTION TAB
# ============================================================

with tab_batch:

    st.subheader(
        "Cohort Optimization Bulk Operations Block"
    )

    st.download_button(
        "⬇️ Download Core CSV Template",

        pd.DataFrame([{
            "country": "Kenya",
            "specimen": "URINE",
            "pathogen": "Escherichia coli",
            "antibiotic": "Ciprofloxacin"
        }]).to_csv(
            index=False
        ).encode(),

        file_name="batch_template.csv",

        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload Target Batch Processing Pipeline Records (CSV)",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            batch_df = pd.read_csv(
                uploaded
            )

            results = []

            for _, row in batch_df.iterrows():

                feat = prepare_features(
                    row["country"],
                    row["specimen"],
                    row["pathogen"],
                    row["antibiotic"]
                )

                p = (
                    model.predict_proba(feat)[0][1]
                    * 100
                )

                label, _ = risk_category(p)

                results.append({

                    **row.to_dict(),

                    "resistance_probability_%":
                        round(p, 1),

                    "risk_category":
                        label

                })

            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Failed parsing uploaded batch context layers: {e}"
            )


# ============================================================
# HISTORY TAB
# ============================================================

with tab_history:

    st.subheader(
        "Session Tracking Logs"
    )

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(
                st.session_state.history
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No query tracking sequences logged "
            "within this context instance loop yet."
        )


# ============================================================
# ABOUT & MODEL CARD TAB
# ============================================================

with tab_about:

    st.subheader(
        "Model Card Specification Ledger"
    )

    if model_card:

        st.json(
            model_card
        )

    else:

        st.markdown(
            "No complete model card architecture "
            "profile template available on system paths."
        )
