"""
generate_research_artifacts_CORRECTED.py — WHO GLASS 2022 real-data version

Same leakage-free evaluation logic as the synthetic-data version (honest
single split matching the real training split, PLUS 5-fold stratified CV),
adapted to the real WHO GLASS 2022 schema: CountryTerritoryArea, Specimen,
PathogenName, AbTargets -> resistant (PercentResistant >= HIGH_RISK_THRESHOLD).

Run from inside code/evaluation/:
    python generate_research_artifacts_CORRECTED.py
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
FEATURE_ORDER = ["CountryTerritoryArea", "Specimen", "PathogenName", "AbTargets"]
HIGH_RISK_THRESHOLD = 20.0  # investigator-defined, not an official WHO cutoff
BEST_PARAMS = dict(n_estimators=300, max_depth=7, learning_rate=0.1)  # from train_model.py's search


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_ORDER + ["PercentResistant"])
    df["resistant"] = (df["PercentResistant"] >= HIGH_RISK_THRESHOLD).astype(int)
    return df


def fit_encoders(df_train):
    encoders = {}
    for col in FEATURE_ORDER:
        le = LabelEncoder()
        le.fit(df_train[col])
        encoders[col] = le
    return encoders


def encode(df, encoders):
    X = df.copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(X[col])
    return X[FEATURE_ORDER]


def make_model(y_train, best_params):
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                              scale_pos_weight=scale, **best_params)


def theoretical_notes(df):
    """Not a strict ceiling (unlike the synthetic dataset, real resistance isn't
    generated from a known fixed probability table) but a useful sanity check:
    accuracy achievable by always predicting each pathogen-antibiotic pair's
    majority class, ignoring country/year entirely."""
    pivot = df.groupby(["PathogenName", "AbTargets"])["resistant"].agg(["mean", "count"])
    pivot["best_const_acc"] = pivot["mean"].apply(lambda p: max(p, 1 - p))
    ceiling = (pivot["best_const_acc"] * pivot["count"]).sum() / pivot["count"].sum()
    print(f"Sanity check — accuracy from pathogen+antibiotic majority-class only "
          f"(no country/year signal used): {ceiling:.3f}")
    return float(ceiling)


def part_a_honest_single_split(df, best_params):
    print("=" * 60); print("PART A - Honest single split (matches real training split)"); print("=" * 60)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    encoders = fit_encoders(df_train)
    X_train = encode(df_train, encoders); y_train = df_train["resistant"]
    X_test = encode(df_test, encoders);  y_test = df_test["resistant"]

    model = make_model(y_train, best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"], output_dict=True)
    brier = brier_score_loss(y_test, y_proba)

    metrics = {
        "split": "single_holdout_test_size_0.3_random_state_42",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "brier_score": float(brier),
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "model_type": type(model).__name__, "features_used": FEATURE_ORDER,
        "target_definition": f"PercentResistant >= {HIGH_RISK_THRESHOLD} (investigator-defined threshold)",
    }
    with open(f"{METRICS_DIR}/metrics_corrected.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                       if k not in ("classification_report", "confusion_matrix")}, indent=2))

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, color="teal", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - WHO GLASS 2022 (held-out test set)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/roc_curve_corrected.png", dpi=150); plt.close()

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - WHO GLASS 2022"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/pr_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix (held-out test set)"); plt.colorbar()
    ticks = ["Low Risk", "High Risk"]; plt.xticks([0, 1], ticks); plt.yticks([0, 1], ticks)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150); plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(mean_pred, frac_pos, "o-", color="#14213D", lw=2, label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency of High Risk")
    plt.title(f"Calibration Curve (Brier = {brier:.3f})"); plt.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIR}/calibration_curve.png", dpi=150); plt.close()

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

    return model, metrics, encoders


def part_b_cross_validation(df, best_params):
    print("=" * 60); print("PART B - 5-fold stratified cross-validation"); print("=" * 60)
    y_all = df["resistant"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, y_all), start=1):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        encoders = fit_encoders(df_tr)
        X_tr = encode(df_tr, encoders); y_tr = df_tr["resistant"]
        X_te = encode(df_te, encoders); y_te = df_te["resistant"]
        model = make_model(y_tr, best_params); model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te); y_proba = model.predict_proba(X_te)[:, 1]

        fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred))
        fold_metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_te, y_proba))
        fold_metrics["pr_auc"].append(average_precision_score(y_te, y_proba))
        fold_metrics["brier"].append(brier_score_loss(y_te, y_proba))
        print(f"Fold {fold}: acc={fold_metrics['accuracy'][-1]:.3f} roc_auc={fold_metrics['roc_auc'][-1]:.3f}")

    summary = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in fold_metrics.items()}
    json.dump(summary, open(f"{METRICS_DIR}/cv_metrics.json", "w"), indent=2)
    print(json.dumps({k: f"{v['mean']:.3f} +/- {v['sd']:.3f}" for k, v in summary.items()}, indent=2))
    return summary


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows. High-risk prevalence: {df['resistant'].mean():.3f}")
    ceiling = theoretical_notes(df)

    model, single_metrics, encoders = part_a_honest_single_split(df, BEST_PARAMS)
    cv_summary = part_b_cross_validation(df, BEST_PARAMS)

    model_card = {
        "model_name": "AMR Predictor — XGBoost Classifier (WHO GLASS 2022 Edition)",
        "model_type": type(model).__name__,
        "version": "2.0-real-data",
        "unit_of_analysis": "Country-year-specimen-pathogen-antibiotic aggregate surveillance "
                             "record (ecological level), NOT individual patient records.",
        "intended_use": "Educational/methodological prototype for country-level AMR surveillance "
                         "risk-pattern analysis. NOT for individual patient diagnosis or treatment.",
        "training_data": {
            "source": "World Health Organization. Global Antimicrobial Resistance and Use "
                       "Surveillance System (GLASS) report: 2022. Geneva: WHO; 2022. "
                       "Licence: CC BY-NC-SA 3.0 IGO. https://iris.who.int/handle/10665/364996 "
                       "Row-level compilation: Quentin Leclerc, "
                       "https://github.com/qleclerc/GLASS2022",
            "n_samples": int(len(df)), "features": FEATURE_ORDER,
            "target": f"resistant (binary: PercentResistant >= {HIGH_RISK_THRESHOLD}, "
                      "investigator-defined threshold, NOT an official WHO risk category)",
            "coverage_caveat": "Compilation includes only countries with continuous 2017-2020 "
                                "data, plus separately all 2019 data; not a complete or fully "
                                "representative sample of all WHO GLASS-enrolled countries."
        },
        "performance_single_holdout": single_metrics,
        "performance_5fold_cv_mean_sd": cv_summary,
        "sanity_check_pathogen_antibiotic_majority_class_accuracy": ceiling,
        "limitations": [
            "Aggregate, country-level surveillance data, not individual patient records; "
            "predictions describe surveillance-reporting combinations, not patients.",
            "The 20% 'High Risk' threshold is investigator-defined, not an official WHO cutoff.",
            "No age, gender, or other patient-level variables are present in this dataset.",
            "Coverage caveat above applies; results should not be read as globally representative.",
        ],
        "authors": ["Chand Javed"],
    }
    with open(f"{METRICS_DIR}/model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    joblib.dump(model, f"{MODELS_DIR}/prediction_model_corrected.pkl")
    joblib.dump(encoders, f"{MODELS_DIR}/feature_encoders.pkl")
    print(f"\nSaved {METRICS_DIR}/model_card.json, {MODELS_DIR}/prediction_model_corrected.pkl, "
          f"{MODELS_DIR}/feature_encoders.pkl")


if __name__ == "__main__":
    main()
