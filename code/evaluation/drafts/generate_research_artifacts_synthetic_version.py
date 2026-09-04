"""
generate_research_artifacts_CORRECTED.py

WHY THIS FILE EXISTS
---------------------
The original generate_research_artifacts.py evaluated prediction_model.pkl
using train_test_split(..., stratify=y), while the model was actually
TRAINED using train_test_split(...) WITHOUT stratify (same random_state,
but stratify changes scikit-learn's internal shuffling, so it produces a
different partition). The result: a large share of the "test" set in the
original script had already been seen during training -> inflated metrics
(91.3% accuracy / 0.961 ROC-AUC) that do not reflect real generalization.

This script fixes that in two independent ways so the numbers you report
are defensible under review:

  (A) HONEST SINGLE SPLIT - retrains fresh, using the *exact* split logic
      used in the real training run (train_model.py / amr_model.ipynb):
      train_test_split(X, y, test_size=0.3, random_state=42) -- NO stratify,
      NO reuse of a model that has seen the test rows. This reproduces the
      genuine ~65% accuracy result.

  (B) 5-FOLD STRATIFIED CROSS-VALIDATION - a more robust, reviewer-proof
      estimate than any single 150-row split. Encoders and model are
      refit inside each fold (no leakage across folds). Report mean +/- SD.

Run this in an environment with xgboost and shap installed:
    pip install xgboost shap scikit-learn pandas numpy matplotlib

Outputs (all leakage-free):
  metrics_corrected.json
  roc_curve_corrected.png
  pr_curve_corrected.png
  confusion_matrix_corrected.png
  calibration_curve_corrected.png
  cv_metrics.json                 <- the numbers to actually put in the paper
  shap_summary.png / shap_bar.png (if shap installed)
  model_card_corrected.json
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

DATA_PATH = "../../data/raw/amr_data.csv"
METRICS_DIR = "../../results/metrics"
FIGURES_DIR = "../../results/figures"
MODELS_DIR = "../../models"
FEATURE_ORDER = ["age", "gender", "specimen", "organism", "antibiotic"]
CATEGORICAL_COLS = ["gender", "specimen", "organism", "antibiotic"]

# Hyperparameters found by the original RandomizedSearchCV
# (amr_model.ipynb: {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.1})
BEST_PARAMS = dict(n_estimators=300, max_depth=7, learning_rate=0.1)


def fit_encoders(df_train):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df_train[col])
        encoders[col] = le
    return encoders


def encode(df, encoders):
    X = df.copy()
    for col, le in encoders.items():
        X[col] = le.transform(X[col])
    return X[FEATURE_ORDER]


def make_model(y_train):
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return xgb.XGBClassifier(
        random_state=42, eval_metric="logloss",
        scale_pos_weight=scale, **BEST_PARAMS
    )


def part_a_honest_single_split(df):
    print("=" * 60)
    print("PART A - Honest single split (matches real training split)")
    print("=" * 60)

    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=42  # NOTE: no stratify -- matches training code
    )
    encoders = fit_encoders(df_train)  # fit ONLY on train -> no leakage
    X_train = encode(df_train, encoders); y_train = df_train["resistant"]
    X_test = encode(df_test, encoders);  y_test = df_test["resistant"]

    model = make_model(y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Sensitive", "Resistant"], output_dict=True)
    brier = brier_score_loss(y_test, y_proba)

    metrics = {
        "split": "single_holdout_test_size_0.3_random_state_42_no_stratify",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "brier_score": float(brier),
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "model_type": type(model).__name__, "features_used": FEATURE_ORDER,
    }
    with open(f"{METRICS_DIR}/metrics_corrected.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                       if k not in ("classification_report", "confusion_matrix")}, indent=2))

    # ROC
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, color="teal", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (leakage-free)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/roc_curve_corrected.png", dpi=150); plt.close()

    # PR
    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (leakage-free)"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/pr_curve_corrected.png", dpi=150); plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix (leakage-free)"); plt.colorbar()
    ticks = ["Sensitive", "Resistant"]; plt.xticks([0, 1], ticks); plt.yticks([0, 1], ticks)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/confusion_matrix_corrected.png", dpi=150); plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(mean_pred, frac_pos, "o-", color="#14213D", lw=2, label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency of resistance")
    plt.title(f"Calibration Curve (Brier = {brier:.3f})"); plt.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/calibration_curve_corrected.png", dpi=150); plt.close()

    # SHAP (global, from this honest fit)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        plt.figure(); shap.summary_plot(sv, X_test, show=False)
        plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
        plt.figure(); shap.summary_plot(sv, X_test, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
        mean_abs = np.abs(np.array(sv)).mean(axis=0).reshape(-1)
        json.dump(dict(zip(FEATURE_ORDER, [float(x) for x in mean_abs])),
                   open(f"{METRICS_DIR}/shap_global_importance.json", "w"), indent=2)
        print("Saved SHAP artifacts.")
    except Exception as e:
        print("SHAP step skipped:", e)

    return model, metrics


def part_b_cross_validation(df):
    print("=" * 60)
    print("PART B - 5-fold stratified cross-validation (report this in the paper)")
    print("=" * 60)
    X_all_raw = df[FEATURE_ORDER + []].copy()
    y_all = df["resistant"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = {k: [] for k in
                     ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all_raw, y_all), start=1):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        encoders = fit_encoders(df_tr)          # refit inside the fold -> no leakage
        X_tr = encode(df_tr, encoders); y_tr = df_tr["resistant"]
        X_te = encode(df_te, encoders); y_te = df_te["resistant"]

        model = make_model(y_tr)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred))
        fold_metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_te, y_proba))
        fold_metrics["pr_auc"].append(average_precision_score(y_te, y_proba))
        fold_metrics["brier"].append(brier_score_loss(y_te, y_proba))
        print(f"Fold {fold}: acc={fold_metrics['accuracy'][-1]:.3f} "
              f"roc_auc={fold_metrics['roc_auc'][-1]:.3f}")

    summary = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "folds": [float(x) for x in v]}
               for k, v in fold_metrics.items()}
    with open(f"{METRICS_DIR}/cv_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: f"{v['mean']:.3f} +/- {v['sd']:.3f}" for k, v in summary.items()}, indent=2))
    return summary


def theoretical_ceiling_check(df):
    """Sanity check against the known data-generating process: since `resistant`
    depends only on organism+antibiotic, no model can beat the accuracy of
    always predicting each pair's majority class. If reported accuracy exceeds
    this, something is leaking."""
    pivot = df.groupby(["organism", "antibiotic"])["resistant"].agg(["mean", "count"])
    pivot["best_const_acc"] = pivot["mean"].apply(lambda p: max(p, 1 - p))
    ceiling = (pivot["best_const_acc"] * pivot["count"]).sum() / pivot["count"].sum()
    print(f"\nTheoretical accuracy ceiling (majority class per organism-antibiotic pair): {ceiling:.3f}")
    return float(ceiling)


def main():
    df = pd.read_csv(DATA_PATH)
    ceiling = theoretical_ceiling_check(df)
    model, single_split_metrics = part_a_honest_single_split(df)
    cv_summary = part_b_cross_validation(df)

    if single_split_metrics["accuracy"] > ceiling + 0.03:
        print("\n*** WARNING: reported accuracy exceeds the theoretical ceiling. "
              "Re-check for leakage before publishing. ***")
    else:
        print("\nSanity check passed: accuracy is consistent with the known "
              "data-generating process (no leakage detected).")

    model_card = {
        "model_name": "AI-Powered Antibiotic Resistance Predictor",
        "model_type": type(model).__name__,
        "version": "1.1.0-corrected",
        "intended_use": "Educational / research prototype for antimicrobial stewardship "
                         "decision support. NOT for direct clinical use.",
        "training_data": {
            "source": "Synthetically generated antibiogram-style dataset (amr_data.csv)",
            "n_samples": int(len(df)), "features": FEATURE_ORDER,
            "target": "resistant (binary: 0 = sensitive, 1 = resistant)"
        },
        "performance_single_holdout": {
            "accuracy": single_split_metrics["accuracy"],
            "roc_auc": single_split_metrics["roc_auc"],
            "pr_auc": single_split_metrics["pr_auc"],
            "f1_score": single_split_metrics["f1_score"],
        },
        "performance_5fold_cv_mean_sd": {
            k: {"mean": v["mean"], "sd": v["sd"]} for k, v in cv_summary.items()
        },
        "theoretical_accuracy_ceiling": ceiling,
        "limitations": [
            "Trained on synthetic, not real hospital antibiogram data.",
            "resistant label depends only on organism x antibiotic in the generating "
            "process; age/gender/specimen carry no true signal by construction.",
            "Small dataset (n=500); wide confidence intervals on all metrics.",
            "Not validated against a real, external hospital cohort.",
            "An earlier evaluation script (generate_research_artifacts.py) reported "
            "inflated metrics (91.3% accuracy) due to a train/test split "
            "mismatch (stratify vs non-stratify) that caused test-set leakage; "
            "those numbers have been superseded by this corrected evaluation."
        ],
        "ethical_considerations": [
            "Model must not be used to make unsupervised treatment decisions.",
            "Predictions should be interpreted alongside culture/susceptibility testing.",
            "Age/gender are included for pattern-learning only; by the data-generating "
            "design they carry no true resistance signal, so any importance assigned "
            "to them by the model reflects noise, not clinical relationships."
        ],
        "authors": ["Chand Javed"],
    }
    with open(f"{METRICS_DIR}/model_card_corrected.json", "w") as f:
        json.dump(model_card, f, indent=2)
    joblib.dump(model, f"{MODELS_DIR}/prediction_model_corrected.pkl")
    print(f"\nSaved {METRICS_DIR}/model_card_corrected.json and {MODELS_DIR}/prediction_model_corrected.pkl")


if __name__ == "__main__":
    main()
