
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import xgboost as xgb



HIGH_RISK_THRESHOLD = 20.0     
DATA_PATH   = r"data\raw\amr_data.csv"    
MODEL_DIR   = "models"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. Update DATA_PATH at the top of "
        f"this script to the full path of your GLASS CSV file."
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[
    "CountryTerritoryArea", "Specimen", "PathogenName",
    "AbTargets", "PercentResistant",
])

df["resistant"] = (df["PercentResistant"] >= HIGH_RISK_THRESHOLD).astype(int)

categorical_cols = [
    "CountryTerritoryArea", "Specimen", "PathogenName", "AbTargets",
]

X = df[categorical_cols].copy()
y = df["resistant"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

print(f"Training rows: {len(X_train):,}   Test rows: {len(X_test):,}")
print(f"High-risk prevalence (train): {y_train.mean():.3f}")

scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

base_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale,
)

search = RandomizedSearchCV(
    base_model,
    param_grid,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
best_model = search.best_estimator_

proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, proba)
pr_auc  = average_precision_score(y_test, proba)
f1      = f1_score(y_test, y_pred)
brier   = brier_score_loss(y_test, proba)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")
print(f"PR-AUC        : {pr_auc:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Brier Score   : {brier:.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

metrics = {
    "accuracy":            float(acc),
    "roc_auc":             float(roc_auc),
    "pr_auc":              float(pr_auc),
    "f1_score":            float(f1),
    "brier_score":         float(brier),
    "high_risk_threshold": HIGH_RISK_THRESHOLD,
    "n_train":             int(len(X_train)),
    "n_test":              int(len(X_test)),
    "best_params":         search.best_params_,
}

with open(os.path.join(METRICS_DIR, "metrics_corrected.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved:", os.path.join(METRICS_DIR, "metrics_corrected.json"))

importances = best_model.feature_importances_
features    = list(X_train.columns)
sorted_idx  = importances.argsort()

plt.figure(figsize=(8, 5))
plt.barh(range(len(features)), importances[sorted_idx], color="teal")
plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance – XGBoost AMR Predictor (WHO GLASS 2022)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
plt.close()


fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#0EA5E9")
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – AMR Predictor")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "roc_curve_corrected.png"), dpi=150)
plt.close()


prec, rec, _ = precision_recall_curve(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AP = {pr_auc:.3f}", color="#10B981")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – AMR Predictor")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pr_curve.png"), dpi=150)
plt.close()


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Low", "High"])
plt.yticks([0, 1], ["Low", "High"])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
plt.close()


json_model_path = os.path.join(MODEL_DIR, "prediction_model_corrected.json")
best_model.save_model(json_model_path)
print("✅ Model saved as JSON:", json_model_path)

ubj_model_path = os.path.join(MODEL_DIR, "prediction_model_corrected.ubj")
best_model.save_model(ubj_model_path)
print("✅ Model saved as UBJ: ", ubj_model_path)

encoders_path = os.path.join(MODEL_DIR, "feature_encoders.pkl")
joblib.dump(encoders, encoders_path)
print("✅ Encoders saved:  ", encoders_path)


model_card = {
    "model_type":       "XGBoost XGBClassifier",
    "xgboost_version":  xgb.__version__,
    "trained_on":       "WHO GLASS 2022 (country-level aggregate surveillance)",
    "target":           f"HighRisk = PercentResistant >= {HIGH_RISK_THRESHOLD}",
    "features":         categorical_cols,
    "metrics":          metrics,
}
with open(os.path.join(METRICS_DIR, "model_card.json"), "w") as f:
    json.dump(model_card, f, indent=2)
print("✅ Model card saved:", os.path.join(METRICS_DIR, "model_card.json"))

print("\n🎉 Training pipeline complete.")