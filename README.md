# AMR Prediction Project — WHO GLASS 2022 Real-Data Edition

## Data replacement notice
This project has been switched from a synthetic per-patient dataset to
**real WHO GLASS 2022 surveillance data**. This is a substantive change, not
a drop-in swap — read this whole notice before using anything in this repo.

### What changed
- **Data**: `data/raw/amr_data.csv` now contains real, verified WHO GLASS 2022
  data (6,402 rows), compiled by a third party from the official WHO GLASS
  2022 report. The old synthetic dataset is preserved at
  `data/archive_synthetic/` for reference — do not delete it, it documents
  the project's earlier phase.
- **Unit of analysis changed**: each row is now a country-year-specimen-
  pathogen-antibiotic AGGREGATE surveillance report, not an individual
  patient. There is no age or gender in this dataset. Any claim framed
  around "predicting whether a patient's infection is resistant" no longer
  describes this model — it now predicts whether a **surveillance-reported
  combination** (e.g., "K. pneumoniae resistance to Ceftriaxone in Kenya")
  falls into a "High Risk" category.
- **Target definition**: `resistant` = 1 if `PercentResistant >= 20.0`.
  **This 20% threshold is a choice made for this analysis — it is NOT an
  official WHO risk classification.** State this explicitly anywhere this
  model's output is described.
- **Features**: `CountryTerritoryArea`, `Specimen`, `PathogenName`,
  `AbTargets` (replacing `age`, `gender`, `specimen`, `organism`,
  `antibiotic`).
- **Model, evaluation script, and app all retrained/rewritten and actually
  re-run** — see CHANGELOG.docx for the full, dated history.
- **`code/app/amr_model_app_prototype.py` has been moved to
  `code/app/drafts/`** — it still used the old age/gender/organism schema and
  would crash against the new model (wrong feature count/names). Only
  `code/app/amr_model_app_enhanced.py` is current and compatible with the
  real-data model.

### Real, verified performance (actually run in this environment — xgboost 3.4.1, shap 0.52.0 — not a placeholder)
Both the single held-out split and 5-fold stratified cross-validation agree
closely, which is itself a good sign there is no leakage (compare to the
earlier synthetic-data leakage episode, where a single-split figure and the
honest figure diverged sharply, 91.3% vs 65.3%):

| Metric | Single held-out split | 5-fold CV (mean ± SD) |
|---|---|---|
| Accuracy | 93.28% | 93.3% ± 0.3% |
| Precision | 0.930 | 0.935 ± 0.005 |
| Recall | 0.940 | 0.933 ± 0.010 |
| F1 | 0.935 | 0.934 ± 0.003 |
| ROC-AUC | 0.977 | 0.977 ± 0.002 |
| PR-AUC | 0.978 | 0.978 ± 0.003 |
| Brier score | 0.053 | 0.053 ± 0.003 |

A sanity check (accuracy from pathogen+antibiotic majority-class alone,
ignoring country/year entirely) gives **79.7%** — meaningfully below the
model's 93.3%, indicating the model is genuinely using country-level
variation, not just memorizing pathogen-antibiotic base rates.

### Known limitations of this data (disclose in any manuscript use)
- The compilation includes only countries with continuous 2017–2020
  reporting, plus separately all 2019 data — countries missing even one
  year (other than 2019) are excluded. This is a real selection bias.
- Aggregate/ecological data only — no patient-level variables exist in this
  dataset (no age, gender, comorbidities, etc.).
- The "High Risk" threshold is investigator-defined, not an official
  WHO classification.

### Data provenance (cite both, precisely)
1. World Health Organization. Global Antimicrobial Resistance and Use
   Surveillance System (GLASS) report: 2022. Geneva: WHO; 2022.
   ISBN 978-92-4-006270-2. Licence: CC BY-NC-SA 3.0 IGO.
   https://iris.who.int/handle/10665/364996
2. Row-level compilation: Quentin Leclerc (2022). Compiled data from WHO
   GLASS 2022 report. GitHub repository: https://github.com/qleclerc/GLASS2022

## Repository structure
```
AMR_Project/
├── data/
│   ├── raw/amr_data.csv          <- REAL WHO GLASS 2022 data (current)
│   └── archive_synthetic/         <- old synthetic data, kept for reference
├── code/
│   ├── data_generation/           <- synthetic-data generation notebooks (archival)
│   ├── training/train_model.py    <- real-data training script (current)
│   ├── evaluation/generate_research_artifacts_CORRECTED.py  <- leakage-free eval (current)
│   ├── app/amr_model_app_enhanced.py  <- Streamlit app, real-data schema (current)
│   └── */drafts/                  <- superseded synthetic-schema versions, kept for reference
├── models/
│   ├── prediction_model_corrected.pkl  <- REAL, retrained model (current) — verified feature names: CountryTerritoryArea, Specimen, PathogenName, AbTargets
│   └── feature_encoders.pkl            <- REAL, retrained encoders (current)
├── results/
│   ├── metrics/   <- metrics_corrected.json, cv_metrics.json, model_card.json (all real)
│   └── figures/   <- ROC/PR/confusion/calibration/SHAP (all real, freshly generated)
└── CHANGELOG.docx
```

## Running it yourself
```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap streamlit joblib
cd code/training && python train_model.py            # retrains on real data
cd ../evaluation && python generate_research_artifacts_CORRECTED.py  # honest eval + figures
cd ../app && streamlit run amr_model_app_enhanced.py  # launch the app
```
This entire sequence was actually executed (not just written) to produce
this package — every number above and every file in results/ and models/
was freshly generated, not carried over or estimated.
