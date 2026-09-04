import streamlit as st
import pandas as pd
import joblib
import numpy as np
model = joblib.load('../../models/prediction_model_corrected.pkl')
encoders = joblib.load('../../models/feature_encoders.pkl')
st.title("🦠 AI‑Powered Antibiotic Resistance Predictor – Prototype for Antimicrobial Stewardship")
st.markdown("""
This tool is a prototype for predicting the resistance for common bacterial infections of an antibiotic.
This is a machine learning model trained on the antibiogram data.
Note: It is not for clinical use. Can only be used**for the educational and research purposes only**
""")
st.sidebar.header("Patient & Sample Information")
organism = st.sidebar.selectbox(
    "Organism (bacteria)",
    ['E. coli', 'K. pneumoniae', 'S. aureus', 'P. aeruginosa']
)
antibiotic = st.sidebar.selectbox(
    "Antibiotic",
    ['Ciprofloxacin', 'Ceftriaxone', 'Meropenem', 'Levofloxacin', 'Amikacin']
)
specimen = st.sidebar.selectbox(
    "Specimen type",
    ['urine', 'blood', 'sputum']
)
gender = st.sidebar.selectbox(
    "Gender",
    ['M', 'F']
)
age = st.sidebar.number_input(
    "Age (years)",
    min_value=1,
    max_value=90,
    value=30
)
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
if st.sidebar.button("Predict Resistance"):
    features = prepare_features(
        organism,
        antibiotic,
        specimen,
        gender,
        age
    )
    proba = model.predict_proba(features)[0]
    resistant_prob = proba[1]
    st.markdown("---")
    st.subheader("Prediction Result")
    prob_percent = resistant_prob * 100
    st.write(f"**Probability of resistance: {prob_percent:.1f}%**")
    st.progress(min(int(prob_percent), 100))
    if prob_percent >= 50:
        st.error("⚠️ High risk of resistance – consider alternative antibiotic.")
    else:
        st.success("✅ Low resistance risk – this antibiotic may be suitable.")
    st.markdown("---")
    st.markdown("---")
    st.subheader("Resistance probability for other antibiotics (same organism)")
    antibiotics_list = ['Ciprofloxacin', 'Ceftriaxone', 'Meropenem', 'Levofloxacin', 'Amikacin']
    probs = []
    for ab in antibiotics_list:
        feat = prepare_features(organism, ab, specimen, gender, age)
        p = model.predict_proba(feat)[0][1] * 100
        probs.append(p)
    chart_data = pd.DataFrame({
        'Antibiotic': antibiotics_list,
        'Resistance (%)': probs
    }).set_index('Antibiotic')
    st.bar_chart(chart_data)
    st.caption("Prediction based on a synthetic-data prototype. Not for clinical use.")
st.markdown("---")
st.markdown("""
- **Project:** AMR Prediction Tool (Prototype)
- **Developer:** Chand Javed
- **GitHub:** [Link to your repository](https://github.com/yourusername/yourrepository)
- **Contact:** [your.email@example.com](chandjavedatrcb17n18@gmail.com)
""")