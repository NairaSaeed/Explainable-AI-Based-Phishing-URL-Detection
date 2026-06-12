"""
train_and_save_best_model.py
============================
Trains and exports ALL models required by the GUI:
  1. Full-feature models (RF + XGBoost) on PhiUSIIL 50-column features
  2. URL-only models (RF + XGBoost) on 27 lexical URL features

Saves to outputs/models/:
  - trained_random_forest_full_features.joblib
  - trained_xgboost_full_features.joblib
  - trained_random_forest_url_only.joblib
  - trained_xgboost_url_only.joblib
  - best_full_feature_model.joblib  (symlink / copy of the winner)
  - best_url_only_model.joblib
  - full_feature_list.json
  - url_only_feature_list.json
  - model_metadata.json
  - X_train_sample_full.npy        (200 rows for LIME background)
  - X_train_sample_url.npy

Usage:
  python train_and_save_best_model.py
"""

import os
import re
import sys
import math
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, precision_score, recall_score)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PRIMARY_CSV  = "datasets/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
MODELS_DIR   = "outputs/models"
RANDOM_STATE = 42
TARGET_PER_CLASS = 60_000

# ──────────────────────────────────────────────────────────────────────────────
# URL-only feature extraction (matches notebook Cell 2 exactly)
# ──────────────────────────────────────────────────────────────────────────────
IP_RE   = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
KW_LIST = ['login', 'verify', 'update', 'secure', 'account', 'bank',
           'paypal', 'wallet', 'invoice', 'password', 'signin', 'confirm']

def _entropy(s: str) -> float:
    s = str(s)
    if not s:
        return 0.0
    cnt = {}
    for c in s:
        cnt[c] = cnt.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in cnt.values())


def url_features(url: str) -> dict:
    """27 lexical features extracted purely from the URL string."""
    url = str(url).strip()
    try:
        p = urlparse(url if '://' in url else 'http://' + url)
    except Exception:
        p = urlparse('')
    host  = p.hostname or ''
    path  = p.path or ''
    query = p.query or ''
    low   = url.lower()

    d = {
        'url_length':                len(url),
        'hostname_length':           len(host),
        'path_length':               len(path),
        'query_length':              len(query),
        'num_dots':                  url.count('.'),
        'num_hyphens':               url.count('-'),
        'num_digits':                sum(c.isdigit() for c in url),
        'num_special_chars':         sum(c in '!@#$%^&*()+=[]{}|;<>,?' for c in url),
        'num_slashes':               url.count('/'),
        'num_subdomains':            max(0, host.count('.') - 1),
        'has_ip_address':            int(bool(IP_RE.search(host))),
        'has_https':                 int(url.startswith('https')),
        'has_at_symbol':             int('@' in url),
        'has_double_slash_redirect': int('//' in path),
        'entropy':                   round(_entropy(url), 4),
    }
    for kw in KW_LIST:
        d[f'count_{kw}'] = low.count(kw)
    return d


def url_feature_df(series):
    return pd.DataFrame([url_features(u) for u in series])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
PHISHING_TOKENS = {'1', 'phishing', 'malicious', 'bad', 'spam',
                   'fake', 'suspicious', '1.0', 'phish'}

def binarize(series):
    def _map(v):
        sv = str(v).strip().lower()
        if sv in PHISHING_TOKENS:
            return 1
        try:
            return 1 if float(sv) == 1.0 else 0
        except ValueError:
            return 0
    return series.map(_map).astype(int)


def evaluate(model, X_te, y_te, name):
    y_pred  = model.predict(X_te)
    y_proba = (model.predict_proba(X_te)[:, 1]
               if hasattr(model, 'predict_proba') else y_pred.astype(float))
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    print(f"  {name:<30} Acc={acc:.6f}  F1={f1:.6f}  AUC={auc:.6f}")
    return {'accuracy': acc, 'f1': f1, 'roc_auc': auc,
            'precision': prec, 'recall': rec}


def build_full_models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2, max_features='sqrt'),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    }


def build_url_models():
    # Use max_depth=8 and balanced class_weight so that no single binary
    # feature (like has_https) dominates at 45%+ importance.
    # max_features='log2' encourages the forest to evaluate more diverse
    # feature combinations per split.
    return {
        'Random Forest URL-only': RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=1, max_features='log2',
            max_depth=10, class_weight='balanced'),
        'XGBoost URL-only': xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading primary dataset …")
    print("=" * 60)
    df_raw = pd.read_csv(PRIMARY_CSV)
    df_raw['_label'] = binarize(df_raw['label'])

    # Balanced subset
    phish = df_raw[df_raw['_label'] == 1]
    legit = df_raw[df_raw['_label'] == 0]
    n_each = min(TARGET_PER_CLASS, len(phish), len(legit))
    print(f"  Balanced subset: {n_each:,} phishing + {n_each:,} legitimate")
    df_work = pd.concat([
        phish.sample(n=n_each, random_state=RANDOM_STATE),
        legit.sample(n=n_each, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ── FULL-FEATURE set ──────────────────────────────────────────────────
    EXCLUDE = {'_label', 'label', 'URL', 'FILENAME', 'Domain', 'TLD', 'Title'}
    FEATURE_COLS = [c for c in df_work.select_dtypes(include=[np.number]).columns
                    if c not in EXCLUDE]
    print(f"\n  Full-feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS[:5]} …")

    X_full = df_work[FEATURE_COLS].fillna(0).values
    y      = df_work['_label'].values

    X_tr_f, X_te_f, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # save background sample for LIME
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_tr_f), size=min(200, len(X_tr_f)), replace=False)
    np.save(os.path.join(MODELS_DIR, 'X_train_sample_full.npy'), X_tr_f[idx])

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Full-feature models")
    print("=" * 60)

    full_results  = {}
    full_metadata = {}

    for name, model in build_full_models().items():
        print(f"\nTraining {name} …")
        model.fit(X_tr_f, y_tr)
        metrics = evaluate(model, X_te_f, y_te, name)
        full_results[name] = (model, metrics)

    # save individual models
    slug_map = {
        'Random Forest': 'trained_random_forest_full_features',
        'XGBoost':       'trained_xgboost_full_features',
    }
    for name, (model, metrics) in full_results.items():
        fname = os.path.join(MODELS_DIR, slug_map[name] + '.joblib')
        joblib.dump(model, fname)
        print(f"  Saved {fname}")
        full_metadata[name] = {
            'name':          name,
            'model_name':    name,
            'model_type':    'full_feature',
            'prediction_mode': 'Full-feature live page extraction',
            'file':          slug_map[name] + '.joblib',
            'label_mapping': {'0': 'LEGITIMATE', '1': 'PHISHING'},
            'classes': [0, 1],
            'feature_count': len(FEATURE_COLS),
            'training_feature_list_file': 'full_feature_list.json',
            **{k: round(v, 8) for k, v in metrics.items()},
        }

    # best full-feature model
    best_full_name = max(full_results, key=lambda n: full_results[n][1]['accuracy'])
    best_full_model = full_results[best_full_name][0]
    joblib.dump(best_full_model, os.path.join(MODELS_DIR, 'best_full_feature_model.joblib'))
    print(f"\n  Best full-feature model: {best_full_name}")

    # ── URL-ONLY set ──────────────────────────────────────────────────────
    url_col = None
    for c in ['URL', 'url', 'WebsiteURL', 'link', 'domain']:
        if c in df_work.columns:
            url_col = c
            break

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: URL-only models")
    print("=" * 60)

    if url_col is None:
        print("  WARNING: No URL column found. Generating URL-only features from full features subset.")
        # Fallback: use a selection of URL-structural features from full-feature set
        URL_FEAT_COLS = [c for c in FEATURE_COLS if c in {
            'URLLength', 'DomainLength', 'IsDomainIP', 'CharContinuationRate',
            'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar',
            'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL',
            'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL',
            'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
            'SpacialCharRatioInURL', 'IsHTTPS', 'URLSimilarityIndex',
            'TLDLegitimateProb', 'URLCharProb',
        }]
        X_url = df_work[URL_FEAT_COLS].fillna(0).values
    else:
        print(f"  URL column: '{url_col}'")
        X_url_df = url_feature_df(df_work[url_col])
        URL_FEAT_COLS = X_url_df.columns.tolist()
        X_url = X_url_df.values

    print(f"  URL-only features ({len(URL_FEAT_COLS)}): {URL_FEAT_COLS[:6]} …")

    X_tr_u, X_te_u, y_tr_u, y_te_u = train_test_split(
        X_url, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    idx_u = rng.choice(len(X_tr_u), size=min(200, len(X_tr_u)), replace=False)
    np.save(os.path.join(MODELS_DIR, 'X_train_sample_url.npy'), X_tr_u[idx_u])

    url_results  = {}
    url_metadata = {}

    for name, model in build_url_models().items():
        print(f"\nTraining {name} …")
        model.fit(X_tr_u, y_tr_u)
        metrics = evaluate(model, X_te_u, y_te_u, name)
        url_results[name] = (model, metrics)

    slug_url_map = {
        'Random Forest URL-only': 'trained_random_forest_url_only',
        'XGBoost URL-only':       'trained_xgboost_url_only',
    }
    for name, (model, metrics) in url_results.items():
        fname = os.path.join(MODELS_DIR, slug_url_map[name] + '.joblib')
        joblib.dump(model, fname)
        print(f"  Saved {fname}")
        url_metadata[name] = {
            'name':          name,
            'model_name':    name,
            'model_type':    'url_only',
            'prediction_mode': 'URL-only fallback model',
            'file':          slug_url_map[name] + '.joblib',
            'label_mapping': {'0': 'LEGITIMATE', '1': 'PHISHING'},
            'classes': [0, 1],
            'feature_count': len(URL_FEAT_COLS),
            'training_feature_list_file': 'url_only_feature_list.json',
            **{k: round(v, 8) for k, v in metrics.items()},
        }

    best_url_name  = max(url_results, key=lambda n: url_results[n][1]['accuracy'])
    best_url_model = url_results[best_url_name][0]
    joblib.dump(best_url_model, os.path.join(MODELS_DIR, 'best_url_only_model.joblib'))
    print(f"\n  Best URL-only model: {best_url_name}")

    # ── Save feature lists ────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, 'full_feature_list.json'), 'w') as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(os.path.join(MODELS_DIR, 'url_only_feature_list.json'), 'w') as f:
        json.dump(URL_FEAT_COLS, f, indent=2)

    # ── Unified metadata ──────────────────────────────────────────────────
    meta = {
        'best_full_feature_model': {
            'name':   best_full_name,
            'file':   'best_full_feature_model.joblib',
            **full_metadata[best_full_name],
        },
        'best_url_only_model': {
            'name':  best_url_name,
            'file':  'best_url_only_model.joblib',
            **url_metadata[best_url_name],
        },
        'all_models': {**full_metadata, **url_metadata},
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # keep legacy best_model_meta.json for any old references
    legacy = {
        'model_name':    best_full_name,
        'accuracy':      full_metadata[best_full_name]['accuracy'],
        'f1':            full_metadata[best_full_name]['f1'],
        'roc_auc':       full_metadata[best_full_name]['roc_auc'],
        'feature_names': FEATURE_COLS,
        'feature_type':  'full URL + host + content features',
    }
    with open(os.path.join(MODELS_DIR, 'best_model_meta.json'), 'w') as f:
        json.dump(legacy, f, indent=2)
    joblib.dump(best_full_model, os.path.join(MODELS_DIR, 'best_model.joblib'))

    print("\n" + "=" * 60)
    print("ALL ARTIFACTS SAVED")
    print("=" * 60)
    for fn in sorted(os.listdir(MODELS_DIR)):
        fp = os.path.join(MODELS_DIR, fn)
        kb = os.path.getsize(fp) / 1024
        print(f"  {fn:<50}  {kb:8.1f} KB")


if __name__ == '__main__':
    main()
