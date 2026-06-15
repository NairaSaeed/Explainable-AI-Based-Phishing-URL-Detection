# Explainable AI-Based Phishing URL Detection

This repository contains the implementation for the paper:

**Explainable AI-Based Phishing URL Detection**
*(ITC-Egypt 2026)*

## Overview

The project builds an explainable, real-time machine learning framework for phishing URL detection using:

- **Random Forest** and **XGBoost** as the primary models (Experiment 1: full-feature)
- **Logistic Regression** and **Linear SVM** as baseline models
- **SHAP** for global feature attribution
- **LIME** for local, instance-level explanation
- An auxiliary **URL-only external validation** experiment (Experiment 2) for cross-dataset generalization

The system follows the paper's architecture exactly:

1. A raw URL is submitted to the dashboard.
2. **Live feature extraction** is attempted: the page is fetched and 50 PhiUSIIL-aligned features (URL lexical + host/domain + HTML/content) are extracted.
3. If the page is fetched successfully with sufficient content coverage, the **Full-feature Random Forest model** (Experiment 1) generates the prediction.
4. If the page fetch fails, times out, is blocked, or content coverage is too sparse, the system **falls back to the URL-only model** (Experiment 2).
5. SHAP and LIME explain the **same model that was used** for prediction.

> **Important**: Detected signals are for analyst interpretation only.
> They do **not** override the trained ML model verdict.

### Decision Logic (paper-aligned)

| Condition | Verdict | Risk |
|-----------|---------|------|
| P(phishing) ≥ 0.75 and P(phishing) ≥ 0.90 | PHISHING | HIGH |
| P(phishing) ≥ 0.75 and P(phishing) < 0.90 | PHISHING | MEDIUM |
| 0.45 ≤ P(phishing) < 0.75 | REVIEW | REVIEW |
| P(phishing) < 0.45 | SAFE | LOW |

### Known Limitations

- **Live fetching**: Some sites block bot requests (403, timeout). These trigger URL-only fallback.
- **JS-rendered sites**: BeautifulSoup cannot parse content delivered by JavaScript. Low content coverage → URL-only fallback.
- **URL-only model generalization**: The URL-only model is trained on the PhiUSIIL lexical distribution and may not generalize perfectly to all real-world URL patterns. It is designed as a fallback for when live content extraction is not possible.
- **Deployment**: Production use requires continuous retraining, threshold calibration per deployment context, and monitoring of model drift.

---

## Repository Contents

Recommended repository structure:

```text
.
├── Phishing_URL_Detection.ipynb
├── README.md
├── requirements.txt
├── outputs/                  # generated after running the notebook
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── explanations/
└── data/                     # local only; do not upload large datasets unless allowed
```

> Note: the `outputs/` and `data/` folders may be generated locally. Large datasets are usually not uploaded directly to GitHub.

---

## Datasets

This project uses two datasets.

### 1. Primary Dataset: PhiUSIIL Phishing URL Dataset

Used for the main full-feature experiment.

Official source:

- UCI Machine Learning Repository:  
  https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset

The main experiment uses a balanced subset of:

- 60,000 phishing URLs
- 60,000 legitimate URLs
- 120,000 total instances

The full-feature experiment uses engineered PhiUSIIL features, including URL-based, host-based, and content-based attributes.

### 2. External Dataset: LegitPhish

Used for the auxiliary URL-only external validation experiment.

Official source:

- Mendeley Data — LegitPhish Dataset:  
  https://data.mendeley.com/datasets/hx4m73v2sf/1

Because external datasets may not contain the same engineered content-based features as PhiUSIIL, the external validation experiment uses URL-only lexical features extracted consistently from both datasets.

Before external evaluation, overlapping URLs between the primary and external datasets are removed to reduce contamination risk.

---

## Experiments

### Experiment 1: Main Full-Feature PhiUSIIL Experiment

This is the main experiment of the paper.

Models:

- Logistic Regression — baseline
- Linear SVM — baseline
- Random Forest — main model
- XGBoost Tuned — main model

Outputs:

- model performance table
- confusion matrices
- feature importance plots
- SHAP global feature importance
- LIME local explanation
- trained model files

### Experiment 2: Auxiliary URL-Only External Validation

This experiment is used only to support cross-dataset generalization analysis.

It does **not** replace the full-feature PhiUSIIL experiment.

Models:

- Logistic Regression
- Random Forest
- XGBoost Tuned

Outputs:

- internal vs external metrics
- external confusion matrix
- overlap-removal summary
- URL-only feature list

---

## Explainability

The notebook generates two levels of explanation:

### SHAP

SHAP is used for global feature attribution. It identifies the most influential features across the test set.

### LIME

LIME is used for local explanation of a single correctly classified phishing instance. It shows which feature conditions contributed to the phishing prediction.

---

## NoOfExternalRef Sensitivity Analysis

The notebook includes a focused analysis of `NoOfExternalRef`, because this feature may be manipulated by attackers.

The analysis includes:

- feature importance rank
- SHAP rank
- distribution by class
- a simple single-feature sensitivity check

This is not a full adversarial robustness evaluation. It is only a preliminary sensitivity analysis.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap lime joblib jupyter
```

---

## Running the Notebook

Open:

```text
Phishing_URL_Detection.ipynb
```

Then update the dataset paths near the top of the notebook:

```python
PRIMARY_DATASET_PATH = "path/to/PhiUSIIL_Phishing_URL_Dataset.csv"
EXTERNAL_DATASET_PATH = "path/to/LegitPhish_or_external_dataset.csv"
OUTPUT_DIR = "outputs"
```

Run the notebook cells from top to bottom.

Generated files will be saved under:

```text
outputs/
├── figures/
├── tables/
├── models/
└── explanations/
```

---

## Main Outputs

The notebook saves results such as:

```text
outputs/tables/results_internal_full_features.csv
outputs/tables/results_internal_url_only.csv
outputs/tables/results_external_url_only.csv
outputs/tables/generalization_results_table.csv
outputs/tables/data_leakage_overlap_summary.csv
outputs/tables/feature_lists.json
outputs/models/trained_random_forest_full_features.joblib
outputs/models/trained_xgboost_full_features.joblib
outputs/models/trained_random_forest_url_only.joblib
outputs/models/trained_xgboost_url_only.joblib
outputs/explanations/lime_local_explanation_full_features.html
outputs/metrics_summary.json
```

Figures are saved in:

```text
outputs/figures/
```

---

## Reproducibility Notes

The notebook uses:

- fixed random seed: `42`
- stratified train/test split
- duplicate removal
- URL/domain overlap checks where URL columns are available
- external overlap removal before external evaluation

The exact results may vary slightly depending on:

- dataset version
- package versions
- operating system
- whether the full external dataset is available locally

---

## Important Notes

- Do not upload large dataset files to GitHub unless the dataset license allows it.
- Prefer linking to the official dataset pages.
- Keep generated outputs if they are needed for paper reproducibility.
- If output folders are large, upload only the most important tables, figures, and model summaries.
- Do not upload private API keys, credentials, local paths, or personal files.

---

## Suggested `.gitignore`

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/
outputs/models/*.joblib
*.pkl
*.pyc
.DS_Store
Thumbs.db
```

If you want to include final paper figures and tables, remove the corresponding `outputs/figures/` or `outputs/tables/` entries from `.gitignore`.

---

## Citation

If you use this project, please cite the associated paper:

```bibtex
@inproceedings{phishing_xai_detection_2026,
  title     = {Explainable AI-Based Phishing URL Detection},
  author    = {Naira Saeed Hassan AL-Battra},
  booktitle = {Proceedings of ITC-Egypt},
  year      = {2026}
}
```

---

## License

Add a license file if you plan to make the repository public. Common choices include:

- MIT License
- Apache License 2.0
- GPL-3.0

If no license is added, others may not have clear permission to reuse the code.

---

## Interactive GUI Dashboard

This project includes an analyst-facing Gradio dashboard that reflects the real-world application of the "Explainable AI-Based Phishing URL Detection" paper. 

### Prediction Modes
The GUI supports two intelligent prediction modes:

1. **Full-feature live extraction**: This mode matches the main paper model. It attempts to fetch the target webpage in real-time. If successful, it extracts all 50 structural and page-content features and runs the **Random Forest Full-feature** model.
2. **URL-only fallback**: Used automatically when webpage fetching fails (e.g., timeouts, blocked connections, or offline pages) or if too few content features are available. It gracefully switches to a backup **Random Forest URL-only** model that evaluates 27 lexical features extracted purely from the URL string.

The dashboard always explicitly displays:
- Which model and prediction mode were used.
- Page fetch status (Success, Timeout, Error, etc.).
- The decision threshold (default 0.75 for High Confidence Phishing).
- A final verdict (`SAFE`, `REVIEW`, or `PHISHING`).

### How to Run the GUI

1. Generate the trained models and metadata artifacts (if you haven't already):
```bash
python train_and_save_best_model.py
```
2. Start the dashboard:
```bash
python gui/app.py
```

### Interpreting the Verdict
- **SAFE** (< 45% confidence): URL structure and content do not exhibit strong phishing signals.
- **REVIEW** (45% - 75% confidence): Elevated probability of phishing. The model is uncertain, and human analyst review is strongly recommended.
- **PHISHING** (>= 75% confidence): High-confidence malicious patterns detected.

### Limitations in Deployment
- **Live Fetching**: Web fetch operations are constrained by timeouts and firewalls. Some malicious sites actively block bots, leading to a fallback URL-only assessment.
- **Data Distribution Shift**: False positives can occur if legitimate pages share structural features with common phishing templates. Threshold calibration is recommended for production deployment.
