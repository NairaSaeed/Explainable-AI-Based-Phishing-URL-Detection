# Explainable AI-Based Phishing URL Detection

A machine learning project for detecting phishing URLs using extracted URL, domain, and page-content features. The project includes trained models and a simple GUI for testing URLs and viewing prediction explanations.

## Features

- Detects phishing and legitimate URLs
- Supports full-feature and URL-only prediction modes
- Uses machine learning models instead of rule-based decisions
- Includes a GUI for entering URLs and viewing results
- Supports explainability using SHAP and LIME
- Includes scripts for training and testing models

## Datasets

This project uses two datasets:

| Dataset | Usage |
|---|---|
| PhiUSIIL Phishing URL Dataset | Used for training and internal testing |
| LegitPhish Dataset | Used for external/generalization testing |

The PhiUSIIL dataset is used to train and evaluate the main models. The LegitPhish dataset is used to test how well the URL-based models generalize to a different dataset.

## Feature Extraction

The project supports two types of feature extraction:

- **Full-feature extraction:** uses URL lexical features, domain/host features, and HTML/content features.
- **URL-only extraction:** uses lexical features extracted from the URL string only.

The full-feature model is used when page content can be extracted. The URL-only model is useful when page content is unavailable or when testing generalization on datasets that contain URLs only.

## Models

The project uses the following machine learning models:

| Model | Role |
|---|---|
| Logistic Regression | Baseline model |
| Linear SVM | Baseline model |
| Random Forest | Main model |
| XGBoost | Main model |

Random Forest and XGBoost are used as the main models. Logistic Regression and Linear SVM are included as baseline models for comparison.

Saved model files are stored in:

```text
outputs/models/
```

## Project Structure

```text
.
├── datasets/                  # Datasets used for training and testing
├── gui/                       # GUI application
├── outputs/                   # Saved models and output files
├── src/                       # Feature extraction, model loading, and prediction code
├── Phishing_URL_Detection.ipynb
├── train_and_save_best_model.py
├── test_predictions.py
├── requirements.txt
└── README.md
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the web interface:

```bash
python gui/app.py
```

Then open the local URL displayed in the terminal.

To test sample predictions:

```bash
python test_predictions.py
```

To retrain the models:

```bash
python train_and_save_best_model.py
```

The trained models will be saved in:

```text
outputs/models/
```

## Label Mapping

The project uses the following label mapping:

| Label | Meaning |
|---|---|
| 0 | Legitimate URL |
| 1 | Phishing URL |

