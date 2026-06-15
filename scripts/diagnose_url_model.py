"""Compare both URL-only models."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import joblib, json, numpy as np
from src.feature_extraction import extract_url_only_features

MODELS_DIR = 'outputs/models'
feats = json.load(open(f'{MODELS_DIR}/url_only_feature_list.json'))

rf  = joblib.load(f'{MODELS_DIR}/trained_random_forest_url_only.joblib')
xgb = joblib.load(f'{MODELS_DIR}/trained_xgboost_url_only.joblib')

test_urls = [
    ("http://192.168.1.1/login/paypal.php",         "PHISHING"),
    ("https://tudasoge.z27.web.core.windows.net/",  "PHISHING"),
    ("https://guide-ledgrr-live.pages.dev/",        "PHISHING"),
    ("https://www.google.com",                      "LEGIT"),
    ("https://edas.info",                           "LEGIT"),
    ("https://www.github.com",                      "LEGIT"),
    ("https://www.microsoft.com",                   "LEGIT"),
]

classes_rf  = list(rf.classes_)
classes_xgb = list(xgb.classes_)
phish_idx_rf  = classes_rf.index(1)
phish_idx_xgb = classes_xgb.index(1)

print(f"{'URL':<50} {'Expected':<10} {'RF prob':<10} {'XGB prob':<10}")
print('-'*82)
for url, expected in test_urls:
    df = extract_url_only_features(url)
    df_aligned = df.reindex(columns=feats, fill_value=0.0)
    vec = df_aligned.iloc[0].values.astype(float).reshape(1,-1)
    pp_rf  = rf.predict_proba(vec)[0][phish_idx_rf]
    pp_xgb = xgb.predict_proba(vec)[0][phish_idx_xgb]
    print(f"{url:<50} {expected:<10} {pp_rf:<10.2%} {pp_xgb:<10.2%}")
