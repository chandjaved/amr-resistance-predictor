# AMR Prediction Project — WHO GLASS 2022 Real-Data Edition

## Data Replacement Notice

This project has been switched from a synthetic per-patient dataset to **real WHO GLASS 2022 surveillance data**. This is a substantive change, not a drop-in swap. Please read this notice before using the repository.

## What Changed

* **Data:** `data/raw/amr_data.csv` now contains real WHO GLASS 2022 surveillance data (6,402 rows), compiled from the official WHO GLASS 2022 report. The old synthetic dataset is preserved at `data/archive_synthetic/` for reference.
* **Unit of analysis:** Each row represents a **country-year-specimen-pathogen-antibiotic aggregate surveillance report**, not an individual patient.
* **Patient-level variables:** There is no age or gender information in this dataset. Therefore, the model does **not** predict whether an individual patient's infection is resistant.
* **Prediction target:** The model predicts whether a **surveillance-reported combination** (for example, *K. pneumoniae* resistance to Ceftriaxone in Kenya) falls into the analysis-defined "High Risk" category.
* **Target definition:** `resistant = 1` when `PercentResistant >= 20.0`.

  **Important:** The 20% threshold is an investigator-defined choice for this analysis. It is **not an official WHO risk classification**.
* **Features:** `CountryTerritoryArea`, `Specimen`, `PathogenName`, and `AbTargets`.
* **Model and evaluation:** The model, evaluation scripts, and Streamlit application were retrained/rewritten for the real-data schema.
* **Application:** `code/app/amr_model_app_enhanced.py` is the current compatible Streamlit application.
* **Archived prototype:** `code/app/drafts/amr_model_app_prototype.py` has been retained for reference because it uses the previous synthetic age/gender/organism schema and is not compatible with the current model.

## Model Performance

The following results were obtained by actually running the model in the development environment using XGBoost 3.4.1 and SHAP 0.52.0.

| Metric      | Single Held-Out Split | 5-Fold CV (Mean ± SD) |
| ----------- | --------------------: | --------------------: |
| Accuracy    |                93.28% |          93.3% ± 0.3% |
| Precision   |                 0.930 |         0.935 ± 0.005 |
| Recall      |                 0.940 |         0.933 ± 0.010 |
| F1          |                 0.935 |         0.934 ± 0.003 |
| ROC-AUC     |                 0.977 |         0.977 ± 0.002 |
| PR-AUC      |                 0.978 |         0.978 ± 0.003 |
| Brier Score |                 0.053 |         0.053 ± 0.003 |

The single held-out split and 5-fold cross-validation results are closely aligned.

A sanity check using only pathogen-antibiotic majority classes, while ignoring country/year information, achieved **79.7% accuracy**. This is substantially below the model's approximately 93.3% accuracy and suggests that the model is using country-level variation rather than relying only on pathogen-antibiotic base rates.

## Known Limitations

These limitations should be disclosed when the dataset or model is used in research or manuscript preparation:

1. **Selection bias:** The compilation includes countries with continuous 2017–2020 reporting, together with separately included 2019 data. Countries missing one or more required years, other than 2019, are excluded.
2. **Aggregate data:** The dataset contains ecological/surveillance-level data rather than patient-level observations. Patient characteristics such as age, gender, comorbidities, and treatment history are not available.
3. **Investigator-defined threshold:** The 20% "High Risk" threshold is defined for this analysis and is **not an official WHO classification**.
4. **No individual-patient prediction:** Results should not be interpreted as predicting resistance for a specific patient.

## Data Provenance

### 1. World Health Organization

World Health Organization. *Global Antimicrobial Resistance and Use Surveillance System (GLASS) report: 2022.* Geneva: WHO; 2022. ISBN 978-92-4-006270-2. Licence: CC BY-NC-SA 3.0 IGO.

Official source:
https://iris.who.int/handle/10665/364996

### 2. Row-Level Data Compilation

Quentin Leclerc (2022). Compiled data from the WHO GLASS 2022 report.

GitHub repository:
https://github.com/qleclerc/GLASS2022

## Repository Structure

```text
AMR_Project/
├── data/
│   ├── raw/
│   │   └── amr_data.csv
│   └── archive_synthetic/
│
├── code/
│   ├── data_generation/
│   │   └── Synthetic-data generation notebooks (archival)
│   │
│   ├── training/
│   │   └── train_model.py
│   │
│   ├── evaluation/
│   │   └── generate_research_artifacts_CORRECTED.py
│   │
│   └── app/
│       ├── amr_model_app_enhanced.py
│       └── drafts/
│
├── models/
│   ├── prediction_model_corrected.pkl
│   └── feature_encoders.pkl
│
├── results/
│   ├── metrics/
│   │   ├── metrics_corrected.json
│   │   ├── cv_metrics.json
│   │   └── model_card.json
│   │
│   └── figures/
│       ├── ROC
│       ├── PR
│       ├── Confusion Matrix
│       ├── Calibration
│       └── SHAP
│
└── CHANGELOG.docx
```

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap streamlit joblib
```

## Training the Model

From the project root:

```bash
cd code/training
python train_model.py
```

## Running the Evaluation

```bash
cd ../evaluation
python generate_research_artifacts_CORRECTED.py
```

This generates the evaluation metrics and research figures.

## Running the Streamlit Application

```bash
cd ../app
streamlit run amr_model_app_enhanced.py
```

The application uses the real-data model and the following features:

* Country/Territory/Area
* Specimen
* Pathogen
* Antibiotic target

## Reproducibility

The training, evaluation, and application workflow was executed in the development environment to produce the current model, metrics, figures, and supporting artifacts.

The repository retains earlier synthetic-data components for historical reference, while the current model and application are based on the WHO GLASS 2022 real-data edition.

## Disclaimer

This project is intended for **research and educational purposes**. The model's "High Risk" classification is based on an investigator-defined 20% resistance threshold and should not be interpreted as an official WHO classification or as a clinical diagnostic decision.

## License and Data Use

The underlying WHO GLASS data are subject to the licensing and attribution requirements specified by the World Health Organization. Users should review the original WHO source and applicable licence terms before redistributing or using the data.
