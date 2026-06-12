"""
test_predictions.py
===================
Validate the full prediction pipeline against all required test URLs.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.predictor import predict_url, get_registry

print("Loading models…")
get_registry()
print()

TEST_CASES = [
    # (url, expected_category)
    ("https://www.google.com",                         "LEGIT"),
    ("https://edas.info",                              "LEGIT"),
    ("https://www.github.com",                         "LEGIT"),
    ("https://www.microsoft.com",                      "LEGIT"),
    ("https://www.queensu.ca",                         "LEGIT"),
    ("https://tudasoge.z27.web.core.windows.net/",     "SUSPICIOUS"),
    ("https://guide-ledgrr-live.pages.dev/",           "SUSPICIOUS"),
    ("http://192.168.1.1/login/paypal.php",            "SUSPICIOUS"),
]

W = 56
for url, category in TEST_CASES:
    res = predict_url(url, "Auto")
    result  = res["final_result"]
    pp      = res["probability_phishing"]
    pl      = res["probability_legitimate"]
    conf    = res["confidence"]
    mode    = res["prediction_mode"]
    fetch   = res["page_fetch_status"]
    feats   = res["feature_set_used"]
    cover   = res["content_coverage"]
    missing = len(res["missing_features"])
    
    print(f"{'─'*W}")
    print(f"URL     : {url}")
    print(f"Category: {category}  |  Result: {result}")
    print(f"  Prob Phishing:   {pp:.2%}")
    print(f"  Prob Legitimate: {pl:.2%}")
    print(f"  Confidence:      {conf:.2%}")
    print(f"  Mode:            {mode}")
    print(f"  Fetch:           {fetch}  |  Coverage: {cover:.0%}")
    print(f"  Feature Set:     {feats}  |  Missing: {missing}")
print(f"{'─'*W}")
