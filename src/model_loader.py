"""
src/model_loader.py
====================
Loads all trained models, explainers, and feature metadata for the GUI.

Models expected under outputs/models/:
  trained_random_forest_full_features.joblib
  trained_xgboost_full_features.joblib
  trained_random_forest_url_only.joblib    (best URL-only)
  full_feature_list.json
  url_only_feature_list.json
  model_metadata.json
  X_train_sample_full.npy   (background for LIME / SHAP)
  X_train_sample_url.npy
"""

from __future__ import annotations

import os
import json
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "models")


def _jload(path: str):
    with open(path) as f:
        return json.load(f)


def _try_load(filename: str):
    p = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p):
        return joblib.load(p)
    return None


def _build_shap(model):
    try:
        import shap
        if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
            return shap.TreeExplainer(model)
    except Exception:
        pass
    return None


def _build_lime(model, background: np.ndarray, feature_names: list):
    try:
        import lime.lime_tabular
        return lime.lime_tabular.LimeTabularExplainer(
            background,
            feature_names=feature_names,
            class_names=["SAFE", "PHISHING"],
            mode="classification",
        )
    except Exception:
        return None


def load_models() -> dict:
    """
    Load and return the model registry.

    Registry structure:
    {
        "<display_name>": {
            "model":      sklearn estimator,
            "features":   list[str],         # exact training column order
            "mode":       "full_feature" | "url_only",
            "meta":       dict,              # accuracy, f1, etc.
            "shap_exp":   TreeExplainer | None,
            "lime_exp":   LimeTabularExplainer | None,
        },
        ...
    }
    """
    # ── Verify prerequisite files ────────────────────────────────────────
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"model_metadata.json not found in:\n  {MODELS_DIR}\n"
            "Run:  python train_and_save_best_model.py"
        )

    meta       = _jload(meta_path)
    full_feats = _jload(os.path.join(MODELS_DIR, "full_feature_list.json"))
    url_feats  = _jload(os.path.join(MODELS_DIR, "url_only_feature_list.json"))
    all_meta   = meta.get("all_models", {})

    # ── Background arrays for LIME ───────────────────────────────────────
    bg_full_path = os.path.join(MODELS_DIR, "X_train_sample_full.npy")
    bg_url_path  = os.path.join(MODELS_DIR, "X_train_sample_url.npy")
    bg_full = np.load(bg_full_path) if os.path.exists(bg_full_path) else None
    bg_url  = np.load(bg_url_path)  if os.path.exists(bg_url_path)  else None

    # ── Load models ──────────────────────────────────────────────────────
    rf_full  = _try_load("trained_random_forest_full_features.joblib")
    xgb_full = _try_load("trained_xgboost_full_features.joblib")
    rf_url   = _try_load("trained_random_forest_url_only.joblib")
    xgb_url  = _try_load("trained_xgboost_url_only.joblib")
    # The 'best' URL-only model (winner of training — may be XGBoost)
    best_url = _try_load("best_url_only_model.joblib")
    best_url_meta = meta.get("best_url_only_model", {})

    entries = []

    if rf_full is not None:
        entries.append((
            "Full-feature Random Forest", rf_full, full_feats, "full_feature",
            all_meta.get("Random Forest", {}), bg_full
        ))
    if xgb_full is not None:
        entries.append((
            "Full-feature XGBoost", xgb_full, full_feats, "full_feature",
            all_meta.get("XGBoost", {}), bg_full
        ))
    # Primary URL-only fallback = the model that won training
    if best_url is not None:
        entries.append((
            "URL-only fallback model", best_url, url_feats, "url_only",
            best_url_meta, bg_url
        ))
    # Also expose RF and XGBoost URL-only variants individually
    if rf_url is not None:
        entries.append((
            "URL-only Random Forest", rf_url, url_feats, "url_only",
            all_meta.get("Random Forest URL-only", {}), bg_url
        ))
    if xgb_url is not None:
        entries.append((
            "URL-only XGBoost", xgb_url, url_feats, "url_only",
            all_meta.get("XGBoost URL-only", {}), bg_url
        ))

    registry: dict = {}
    for name, model, feats, mode, m_meta, bg in entries:
        shap_exp = _build_shap(model)
        lime_exp = _build_lime(model, bg, feats) if bg is not None else None
        registry[name] = {
            "model":    model,
            "features": feats,
            "mode":     mode,
            "meta":     m_meta,
            "shap_exp": shap_exp,
            "lime_exp": lime_exp,
        }

    print(f"[model_loader] Loaded {len(registry)} model(s): {list(registry.keys())}")
    return registry
