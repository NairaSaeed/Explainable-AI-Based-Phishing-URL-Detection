"""
src/predictor.py
================
Paper-aligned prediction pipeline for the Phishing URL Detection system.

predict_url(url, mode) -> dict

Auto-mode logic (mirrors the paper):
  1. Attempt full-feature live page extraction.
  2. If page is fetched AND content coverage >= MIN_CONTENT_COVERAGE:
       → use Full-feature Random Forest (Experiment 1 of paper)
  3. Otherwise:
       → fall back to URL-only Random Forest (Experiment 2 / external validation)

Decision thresholds (paper-aligned):
  P(phishing) >= 0.75  → PHISHING  (HIGH if >= 0.90, else MEDIUM)
  0.45 <= P < 0.75     → REVIEW
  P < 0.45             → SAFE

IMPORTANT: detected_signals are for ANALYST DISPLAY ONLY.
           They NEVER override the ML model prediction.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from urllib.parse import urlparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import (
    extract_full_features,
    extract_url_only_features,
    get_detected_signals,
)
from src.model_loader import load_models

# ── Thresholds ─────────────────────────────────────────────────────────────
PHISHING_THRESHOLD = 0.75
REVIEW_THRESHOLD   = 0.45

# Cached registry (lazy-loaded once)
_REGISTRY: dict | None = None


def get_registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_models()
    return _REGISTRY


# ── Probability helper ─────────────────────────────────────────────────────
def _phishing_prob(model, vec_2d: np.ndarray) -> tuple[float, float, int]:
    """
    Return (prob_phishing, prob_safe, phishing_class_index).
    Correctly handles any class ordering from model.classes_.
    """
    proba   = model.predict_proba(vec_2d)[0]
    classes = list(model.classes_)
    try:
        phish_idx = classes.index(1)
    except ValueError:
        phish_idx = 1          # fallback: assume index 1 is phishing
    safe_idx = 1 - phish_idx  # the other class
    return float(proba[phish_idx]), float(proba[safe_idx]), phish_idx


# ── Explainability ─────────────────────────────────────────────────────────
def _shap_explanation(entry: dict, vec_2d: np.ndarray, feat_names: list) -> list[str]:
    shap_exp = entry.get("shap_exp")
    if shap_exp is None:
        return ["Explanation unavailable for this model/run."]
    try:
        sv = shap_exp.shap_values(vec_2d)
        # TreeExplainer returns list[ndarray] for multi-class; take class-1 (phishing)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv_flat = np.array(sv).flatten()
        top = sorted(zip(feat_names, sv_flat), key=lambda x: abs(x[1]), reverse=True)[:10]
        lines = [
            "Feature importance (SHAP — phishing direction)",
            "─" * 52,
        ]
        for fname, val in top:
            direction = "⬆ phishing" if val > 0 else "⬇ safe"
            lines.append(f"  {fname:<32} {val:+.5f}  {direction}")
        return lines
    except Exception as exc:
        return [f"Explanation unavailable for this model/run. ({exc})"]


def _lime_explanation(
    entry: dict, vec: np.ndarray, feat_names: list, pred_label: int
) -> list[str]:
    lime_exp = entry.get("lime_exp")
    model    = entry["model"]
    if lime_exp is None:
        return ["Explanation unavailable for this model/run."]
    try:
        exp = lime_exp.explain_instance(
            vec,
            model.predict_proba,
            num_features=10,
            top_labels=2,
        )
        lime_list = exp.as_list(label=pred_label)
        lines = [
            f"Local explanation (LIME — label={pred_label}: "
            f"{'PHISHING' if pred_label else 'SAFE'})",
            "─" * 52,
        ]
        for cond, weight in lime_list:
            direction = "⬆ phishing" if weight > 0 else "⬇ safe"
            lines.append(f"  {cond:<40} {weight:+.5f}  {direction}")
        return lines
    except Exception as exc:
        return [f"Explanation unavailable for this model/run. ({exc})"]


# ── Main prediction function ────────────────────────────────────────────────
def predict_url(url_raw: str, mode: str = "Auto") -> dict:
    """
    Run the full prediction pipeline and return a result dictionary.

    Parameters
    ----------
    url_raw : str  – raw URL entered by the user
    mode    : str  – one of: "Auto", "Full-feature Random Forest",
                              "Full-feature XGBoost", "URL-only fallback model"

    Returns
    -------
    dict with all fields required by the GUI.
    """
    # ── 1. Normalise URL ───────────────────────────────────────────────
    url = url_raw.strip()
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc or (parsed.path.split('/')[0])

    registry = get_registry()
    auto_mode = (mode == "Auto")

    # ── 2. Feature extraction & model selection ────────────────────────
    df_feats:    pd.DataFrame
    fetch_status: str = "not_attempted"
    mode_used:   str  = "url_only"
    entry:       dict | None = None
    content_coverage: float = 0.0

    want_full = auto_mode or mode in ("Full-feature Random Forest", "Full-feature XGBoost")

    if want_full:
        df_full, meta = extract_full_features(url, fetch_timeout=6)
        fetch_status     = meta["fetch_status"]
        content_coverage = meta["content_coverage"]
        is_full          = meta["is_full_extract"]

        if is_full:
            # Full-feature extraction succeeded — pick the requested or default model
            if mode == "Full-feature XGBoost":
                entry_key = "Full-feature XGBoost"
            else:
                entry_key = "Full-feature Random Forest"
            entry      = registry.get(entry_key) or registry.get("Full-feature Random Forest")
            mode_used  = "full_feature"
            df_feats   = df_full
        # else: fall through to URL-only below

    if entry is None:
        # URL-only fallback (or explicit URL-only request)
        # Use the best URL-only model (XGBoost URL-only from metadata)
        entry_key = "URL-only fallback model"
        # Also try XGBoost variant which tends to be better calibrated
        entry = registry.get(entry_key) or registry.get("URL-only XGBoost")
        mode_used  = "url_only"
        df_feats   = extract_url_only_features(url)
        if fetch_status == "not_attempted":
            fetch_status = "not_required"

    if entry is None:
        raise RuntimeError(
            "No model loaded. "
            "Run:  python train_and_save_best_model.py"
        )

    model         = entry["model"]
    expected_feats = entry["features"]   # exact training column order
    m_meta        = entry.get("meta", {})

    # ── 3. Strict feature alignment ────────────────────────────────────
    extracted_cols = list(df_feats.columns)
    missing = [f for f in expected_feats if f not in extracted_cols]
    extra   = [f for f in extracted_cols  if f not in expected_feats]

    # Reindex to exact training column order; fill missing with 0.0
    df_aligned = df_feats.reindex(columns=expected_feats, fill_value=0.0)

    vec    = df_aligned.iloc[0].values.astype(float)
    vec_2d = vec.reshape(1, -1)

    # ── 4. ML Prediction (probabilities only) ─────────────────────────
    prob_phishing, prob_safe, phish_idx = _phishing_prob(model, vec_2d)

    # ── 5. Threshold-based verdict (paper-aligned, ML-only) ───────────
    if prob_phishing >= PHISHING_THRESHOLD:
        final_result = "PHISHING"
        risk_level   = "HIGH" if prob_phishing >= 0.90 else "MEDIUM"
        confidence   = prob_phishing
    elif prob_phishing >= REVIEW_THRESHOLD:
        final_result = "REVIEW"
        risk_level   = "REVIEW"
        confidence   = max(prob_phishing, prob_safe)
    else:
        final_result = "SAFE"
        risk_level   = "LOW"
        confidence   = prob_safe

    pred_label = 1 if final_result == "PHISHING" else 0

    # ── 6. Analyst signals (display only, NO verdict override) ─────────
    feats_dict  = df_aligned.iloc[0].to_dict()
    signals     = get_detected_signals(feats_dict, mode_used)

    summary = {
        "PHISHING": "High-confidence malicious patterns detected.",
        "REVIEW":   "Uncertain prediction. Analyst review recommended.",
        "SAFE":     "No strong phishing indicators detected by the model.",
    }[final_result]

    # ── 7. Explanations ───────────────────────────────────────────────
    shap_out = _shap_explanation(entry, vec_2d, expected_feats)
    lime_out = _lime_explanation(entry, vec, expected_feats, pred_label)

    # ── 8. Diagnostics ─────────────────────────────────────────────────
    top_vals = sorted(
        [(f, feats_dict.get(f, 0.0)) for f in expected_feats],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]

    diagnostics = {
        "model_file":          m_meta.get("file", "N/A"),
        "model_classes":       list(model.classes_),
        "phishing_class_index": phish_idx,
        "decision_threshold":  f">={PHISHING_THRESHOLD:.0%} PHISHING  |  >={REVIEW_THRESHOLD:.0%} REVIEW  |  else SAFE",
        "vec_shape":           list(vec_2d.shape),
        "top_extracted":       top_vals,
    }

    return {
        "url":                    url,
        "domain":                 domain,
        "selected_model":         (
            m_meta.get("name")
            or m_meta.get("model_name")
            or ("Full-feature Random Forest" if mode_used == "full_feature"
                else "URL-only fallback model")
        ),
        "prediction_mode":        (
            "Full-feature live page extraction"
            if mode_used == "full_feature" else
            "URL-only fallback"
        ),
        "page_fetch_status":      fetch_status,
        "feature_set_used":       (
            f"{len(expected_feats)} full features"
            if mode_used == "full_feature" else
            f"{len(expected_feats)} URL-only features"
        ),
        "expected_feature_count": len(expected_feats),
        "extracted_feature_count":len(extracted_cols),
        "missing_features":       missing,
        "extra_features":         extra,
        "probability_phishing":   prob_phishing,
        "probability_legitimate": prob_safe,
        "final_result":           final_result,
        "risk_level":             risk_level,
        "confidence":             confidence,
        "detected_signals":       signals,
        "summary":                summary,
        "shap_explanation":       shap_out,
        "lime_explanation":       lime_out,
        "diagnostics":            diagnostics,
        "model_accuracy":         m_meta.get("accuracy", 0.0),
        "model_f1":               m_meta.get("f1", 0.0),
        "content_coverage":       content_coverage,
    }
