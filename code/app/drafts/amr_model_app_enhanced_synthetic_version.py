import io
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
model = joblib.load('../../models/prediction_model_corrected.pkl')
encoders = joblib.load('../../models/feature_encoders.pkl')
def prepare_features(organism, antibiotic, specimen, gender, age):
    input_dict = {
        'organism': organism,
        'antibiotic': antibiotic,
        'specimen': specimen,
        'gender': gender,
        'age': age
    }
    input_df = pd.DataFrame([input_dict])
    input_df['organism'] = encoders['organism'].transform(input_df['organism'])
    input_df['antibiotic'] = encoders['antibiotic'].transform(input_df['antibiotic'])
    input_df['specimen'] = encoders['specimen'].transform(input_df['specimen'])
    input_df['gender'] = encoders['gender'].transform(input_df['gender'])
    input_df = input_df[['age', 'gender', 'specimen', 'organism', 'antibiotic']]
    return input_df
ORGANISMS = ['E. coli', 'K. pneumoniae', 'S. aureus', 'P. aeruginosa']
ANTIBIOTICS = ['Ciprofloxacin', 'Ceftriaxone', 'Meropenem', 'Levofloxacin', 'Amikacin']
SPECIMENS = ['urine', 'blood', 'sputum']
GENDERS = ['M', 'F']
st.set_page_config(
    page_title="AMR Predictor — Research Edition",
    page_icon="🦠",
    layout="wide",
)
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
/* Default text styling - inherits from Streamlit theme */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
/* Sidebar Specific Styling */
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
/* High-Contrast Accessible Hero Module */
.app-header {
    background: linear-gradient(135deg, #1E293B 0%, #334155 100%) !important;
    padding: 2.25rem !important;
    border-radius: 12px;
    margin-bottom: 1.5rem !important;
    border-left: 6px solid #0EA5E9;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
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
/* Hover effect for text elements */
.app-header h1:hover {
    transform: scale(1.02);
    transition: transform 0.2s ease;
}
.app-header p:hover {
    color: #E2E8F0 !important;
    transition: color 0.2s ease;
}
/* Badge styling */
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
/* Card styling */
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
/* Risk chip styling */
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
/* Tabs readability and hover */
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
/* Subheader hover effect */
.stSubheader:hover {
    color: #0EA5E9 !important;
    transition: color 0.2s ease;
}
/* Metric cards hover */
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
/* Button hover effects */
.stButton button {
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}
/* Expander hover */
.streamlit-expanderHeader:hover {
    color: #0EA5E9 !important;
}
/* DataFrame hover */
div[data-testid="stDataFrame"] td:hover {
    background: #F1F5F9 !important;
    transition: background 0.2s ease;
}
/* Download button hover */
div[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}
/* Sidebar button hover */
section[data-testid="stSidebar"] .stButton button:hover {
    background: #0EA5E9 !important;
    color: white !important;
}
/* Information box hover */
.stAlert:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: box-shadow 0.3s ease;
}
/* Text hover effects for general content */
p:hover, span:hover, li:hover {
    transition: opacity 0.2s ease;
}
/* Ensure text is visible on all backgrounds */
.stMarkdown {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)
def gauge_svg(prob_percent: float, color_hex: str) -> str:
    """Signature component: arc gauge framework rendering absolute probability values."""
    radius = 70
    circumference = np.pi * radius
    offset = circumference * (1 - prob_percent / 100)
    return f"""
    <div style="text-align:center;margin:0;padding:0;width:100%;transition: all 0.3s ease;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
      <svg width="180" height="110" viewBox="0 0 180 110" style="max-width:100%;display:block;margin:0 auto;">
        <path d="M 20 90 A {radius} {radius} 0 0 1 160 90" fill="none"
              stroke="#E2E8F0" stroke-width="14" stroke-linecap="round"/>
        <path d="M 20 90 A {radius} {radius} 0 0 1 160 90" fill="none"
              stroke="{color_hex}" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{circumference}" stroke-dashoffset="{offset}">
          <animate attributeName="stroke-dashoffset" from="{circumference}" to="{offset}" dur="1s" fill="freeze"/>
        </path>
        <text x="90" y="80" text-anchor="middle" font-family="IBM Plex Mono, monospace"
              font-size="26" font-weight="600" fill="{color_hex}">''' + f"{prob_percent:.1f}" + '''%</text>
        <text x="90" y="100" text-anchor="middle" font-family="Inter, sans-serif"
              font-size="11" fill="#64748B" font-weight="500">resistance probability</text>
      </svg>
    </div>
    """
def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
metrics = _load_json("../../results/metrics/metrics_corrected.json")
model_card = _load_json("../../results/metrics/model_card.json")
shap_importance = _load_json("../../results/metrics/shap_global_importance.json")
if "history" not in st.session_state:
    st.session_state.history = []
def risk_category(prob_percent: float):
    if prob_percent < 30:
        return "Low Risk", "green"
    elif prob_percent < 60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"
def clinical_interpretation(organism, antibiotic, prob_percent, alt_probs):
    label, _ = risk_category(prob_percent)
    best_alt = min(alt_probs, key=lambda x: x[1])
    text = (
        f"The model estimates a **{prob_percent:.1f}%** probability that this "
        f"**{organism}** isolate is resistant to **{antibiotic}** ({label.lower()}). "
    )
    if best_alt[0] != antibiotic and best_alt[1] < prob_percent - 5:
        text += (
            f"Among the antibiotics considered, **{best_alt[0]}** shows the lowest "
            f"predicted resistance probability ({best_alt[1]:.1f}%) for this organism, "
            "and may warrant discussion as an alternative — pending culture and "
            "susceptibility confirmation."
        )
    else:
        text += "No clearly lower-risk alternative was identified among the antibiotics considered."
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
        out = pd.DataFrame({
            "Feature": features_df.columns,
            "Value": features_df.iloc[0].values,
            "SHAP contribution": sv,
        }).sort_values("SHAP contribution", key=abs, ascending=False)
        return out
    except Exception:
        return None
def build_pdf_report(patient, prob_percent, label, interpretation) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    import textwrap
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "AMR Resistance Prediction Report (Research Prototype)")
    y -= 1 * cm
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
    c.drawString(2.2 * cm, y, f"Predicted resistance probability: {prob_percent:.1f}%  ({label})")
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
  <h1>🦠 AI-Powered Antibiotic Resistance Predictor</h1>
  <p>Research prototype for antimicrobial stewardship decision support · Educational &amp; research use only</p>
</div>
""", unsafe_allow_html=True)
_version = "1.0.0"
_n = "—"
st.markdown(f"""
<div class="badge-row">
  <span class="badge">🧠 SHAP Explainable AI</span>
  <span class="badge">📄 Model Card v{_version}</span>
  <span class="badge">🧪 Synthetic Data (n={_n})</span>
  <span class="badge">📊 Performance Telemetry Active</span>
  <span class="badge">⚕️ Research Blueprint</span>
</div>
""", unsafe_allow_html=True)
with st.expander("ℹ️ About this tool, its data, and its limitations", expanded=False):
    st.markdown(
        "This tool predicts the probability of antimicrobial resistance from "
        "patient/specimen/organism/antibiotic features, trained on a synthetic "
        "antibiogram-style dataset. **Not for clinical use.**"
    )
with st.sidebar:
    st.header("Patient & Sample Information") 
    if st.button("🎲 Load Random Sample Patient", use_container_width=True):
        rng = np.random.default_rng()
        st.session_state["_sample"] = {
            "organism": rng.choice(ORGANISMS),
            "antibiotic": rng.choice(ANTIBIOTICS),
            "specimen": rng.choice(SPECIMENS),
            "gender": rng.choice(GENDERS),
            "age": int(rng.integers(1, 91)),
        }
    sample = st.session_state.get("_sample", {})
    organism = st.selectbox("Organism (bacteria)", ORGANISMS,
                            index=ORGANISMS.index(sample["organism"]) if sample.get("organism") in ORGANISMS else 0)
    antibiotic = st.selectbox("Antibiotic", ANTIBIOTICS,
                               index=ANTIBIOTICS.index(sample["antibiotic"]) if sample.get("antibiotic") in ANTIBIOTICS else 0)
    specimen = st.selectbox("Specimen type", SPECIMENS,
                             index=SPECIMENS.index(sample["specimen"]) if sample.get("specimen") in SPECIMENS else 0)
    gender = st.selectbox("Gender", GENDERS,
                           index=GENDERS.index(sample["gender"]) if sample.get("gender") in GENDERS else 0)
    age = st.number_input("Age (years)", min_value=1, max_value=90, value=int(sample.get("age", 30)))

    predict_clicked = st.button("Predict Resistance", type="primary", use_container_width=True) 
    st.markdown("---")
    st.caption(
        "Educational/research prototype only. Not for clinical use. "
        "See 'About & Model Card' tab for full disclaimers."
    )
tab_predict, tab_perf, tab_shap, tab_batch, tab_history, tab_about = st.tabs(
    ["🔬 Predict", "📊 Model Performance", "🧠 Explainability", "📁 Batch Prediction", "🕒 History", "📖 About & Model Card"]
)
with tab_predict:
    if predict_clicked:
        features = prepare_features(organism, antibiotic, specimen, gender, age)
        proba = model.predict_proba(features)[0]
        resistant_prob = proba[1]
        prob_percent = resistant_prob * 100
        label, color = risk_category(prob_percent)
        color_hex = {"green": "#10B981", "orange": "#F59E0B", "red": "#EF4444"}[color]
        risk_class = {"green": "risk-low", "orange": "risk-moderate", "red": "risk-high"}[color]      
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Result")
            st.markdown(gauge_svg(prob_percent, color_hex), unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:center;margin-top:0.6rem;">'
                f'<span class="risk-chip {risk_class}">{label}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Resistance Context Matrix")
            probs = []
            for ab in ANTIBIOTICS:
                feat = prepare_features(organism, ab, specimen, gender, age)
                p = model.predict_proba(feat)[0][1] * 100
                probs.append((ab, p))
            chart_data = pd.DataFrame(probs, columns=["Antibiotic", "Resistance (%)"]).set_index("Antibiotic")
            st.bar_chart(chart_data, color="#0EA5E9")
            st.markdown('</div>', unsafe_allow_html=True)         
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🩺 Clinical Interpretation Summary")
        interpretation = clinical_interpretation(organism, antibiotic, prob_percent, probs)
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
            "age": age, "gender": gender, "specimen": specimen,
            "organism": organism, "antibiotic": antibiotic,
            "resistance_probability_%": round(prob_percent, 1),
            "risk_category": label,
        })       
        st.subheader("📥 Export Pipeline Data")
        patient = {"Age": age, "Gender": gender, "Specimen": specimen,
                   "Organism": organism, "Antibiotic": antibiotic}
        csv_bytes = pd.DataFrame([{**patient, "Resistance_Probability_%": round(prob_percent, 1),
                                    "Risk_Category": label}]).to_csv(index=False).encode()      
        colA, colB = st.columns(2)
        with colA:
            st.download_button("⬇️ Download as CSV Matrix", csv_bytes, file_name="amr_prediction.csv", mime="text/csv")
        with colB:
            try:
                pdf_bytes = build_pdf_report(patient, prob_percent, label, interpretation)
                st.download_button("⬇️ Download PDF Report Artifact", pdf_bytes, file_name="amr_prediction_report.pdf", mime="application/pdf")
            except Exception as e:
                st.caption(f"PDF engine failed context initialization: {e}")
    else:
        st.info("Set patient/sample details in the sidebar and click **Predict Resistance** to begin.")
with tab_perf:
    st.subheader("Held-out Test Performance Configuration")
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy Matrix", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")
        c3.metric("PR-AUC Score", f"{metrics['pr_auc']:.3f}")
        c4.metric("F1-Score Metrics", f"{metrics['f1_score']:.3f}")
        if "brier_score" in metrics:
            c5.metric("Brier Score", f"{metrics['brier_score']:.3f}")      
        st.markdown("---")       
        colA, colB, colC = st.columns(3)
        for col, img, cap in [
            (colA, "../../results/figures/roc_curve_corrected.png", "ROC Curve Map"),
            (colB, "../../results/figures/pr_curve.png", "Precision–Recall Curve Analysis"),
            (colC, "../../results/figures/confusion_matrix.png", "Confusion Matrix Breakdown"),
        ]:
            try:
                col.image(img, caption=cap, use_container_width=True)
            except Exception:
                col.caption(f"{cap} asset missing from disk volume paths.")
    else:
        st.warning("Telemetry descriptor structure metrics.json not located.")
with tab_shap:
    st.subheader("Global Explainer Feature Profiles (SHAP Engine)")
    colA, colB = st.columns(2)
    try:
        colA.image("../../results/figures/shap_bar.png", caption="Mean |SHAP value| Performance Graph", use_container_width=True)
    except Exception:
        colA.caption("shap_bar.png visualization target absent from disk path storage.")
    try:
        colB.image("../../results/figures/shap_summary.png", caption="SHAP Positional Summary Distribution Diagram", use_container_width=True)
    except Exception:
        colB.caption("shap_summary.png visualization target absent from disk path storage.")
with tab_batch:
    st.subheader("Cohort Optimization Bulk Operations Block")
    st.download_button(
        "⬇️ Download Core CSV Template",
        pd.DataFrame([{"age": 45, "gender": "F", "specimen": "urine", "organism": "E. coli", "antibiotic": "Ciprofloxacin"}]).to_csv(index=False).encode(),
        file_name="batch_template.csv", mime="text/csv",
    )
    uploaded = st.file_uploader("Upload Target Batch Processing Pipeline Records (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            results = []
            for _, row in batch_df.iterrows():
                feat = prepare_features(row["organism"], row["antibiotic"], row["specimen"], row["gender"], row["age"])
                p = model.predict_proba(feat)[0][1] * 100
                label, _ = risk_category(p)
                results.append({**row.to_dict(), "resistance_probability_%": round(p, 1), "risk_category": label})
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Failed parsing uploaded batch context layers: {e}")
with tab_history:
    st.subheader("Session Tracking Logs")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    else:
        st.info("No query tracking sequences logged within this context instance loop yet.")
with tab_about:
    st.subheader("Model Card Specification Ledger")
    if model_card:
        st.json(model_card)
    else:
        st.markdown("No complete model card architecture profile template available on system paths.")