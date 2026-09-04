"""
train_model.py — WHO GLASS 2022 real-data version

Replaces the synthetic-data version (see code/training/drafts/train_model_synthetic_version.py).
Trains an XGBoost classifier on real WHO GLASS 2022 surveillance data to
predict whether a country-year-specimen-pathogen-antibiotic surveillance
record represents a "high resistance" combination.

IMPORTANT — read before using downstream:
  - Unit of analysis is a COUNTRY-LEVEL AGGREGATE surveillance report, not an
    individual patient. There is no age/gender/per-case data in this dataset.
  - "High resistance" (PercentResistant >= HIGH_RISK_THRESHOLD) is a threshold
    chosen for this analysis, NOT an official WHO risk category. State this
    explicitly in any manuscript text that references it.
  - Source: World Health Organization. Global Antimicrobial Resistance and Use
    Surveillance System (GLASS) report: 2022. Geneva: WHO; 2022.
    Licence: CC BY-NC-SA 3.0 IGO. https://iris.who.int/handle/10665/364996
    Row-level compilation: Quentin Leclerc, https://github.com/qleclerc/GLASS2022
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import xgboost as xgb

HIGH_RISK_THRESHOLD = 20.0  # investigator-defined for this analysis, not an official WHO cutoff

df = pd.read_csv('amr_data.csv')
df = df.dropna(subset=['CountryTerritoryArea', 'Specimen', 'PathogenName', 'AbTargets', 'PercentResistant'])
df['resistant'] = (df['PercentResistant'] >= HIGH_RISK_THRESHOLD).astype(int)

categorical_cols = ['CountryTerritoryArea', 'Specimen', 'PathogenName', 'AbTargets']
X = df[categorical_cols].copy()
y = df['resistant']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scale = (y_train == 0).sum() / (y_train == 1).sum()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
base_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale
)
search = RandomizedSearchCV(
    base_model,
    param_grid,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Low Risk', 'High Risk']
))
importances = best_model.feature_importances_
features = X_train.columns
sorted_idx = importances.argsort()
plt.figure(figsize=(8, 5))
plt.barh(range(len(features)), importances[sorted_idx], color='teal')
plt.yticks(range(len(features)), features[sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance – XGBoost AMR Predictor (WHO GLASS 2022)")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("Feature importance plot saved as 'feature_importance.png'")
joblib.dump(best_model, 'amr_model.pkl')
joblib.dump(encoders, 'feature_encoders.pkl')
print("Model saved: amr_model.pkl")
print("Encoders saved: feature_encoders.pkl")
