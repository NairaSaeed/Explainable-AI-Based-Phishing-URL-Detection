import joblib
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('outputs/models/best_model.joblib')
with open('outputs/models/best_model_meta.json') as f:
    meta = json.load(f)

feat_names = meta['feature_names']
print('Model type:', type(model).__name__)
print('Classes:', model.classes_)
print()

# Test 1: All zeros (page fetch completely fails)
vec_zeros = np.zeros((1, 50))
p = model.predict_proba(vec_zeros)[0]
phishing_idx = list(model.classes_).index(1)
print('== All-zeros vector ==')
print('  predict_proba:', p)
print('  P(phishing):', p[phishing_idx])
print()

# Test 2: URL-only features for google.com, content zeros
google_vec = {
    'URLLength': 22, 'DomainLength': 10, 'IsDomainIP': 0, 'URLSimilarityIndex': 100.0,
    'CharContinuationRate': 0.5, 'TLDLegitimateProb': 0.5, 'URLCharProb': 0.5,
    'TLDLength': 3, 'NoOfSubDomain': 0, 'HasObfuscation': 0, 'NoOfObfuscatedChar': 0,
    'ObfuscationRatio': 0.0, 'NoOfLettersInURL': 16, 'LetterRatioInURL': 0.73,
    'NoOfDegitsInURL': 0, 'DegitRatioInURL': 0.0, 'NoOfEqualsInURL': 0,
    'NoOfQMarkInURL': 0, 'NoOfAmpersandInURL': 0, 'NoOfOtherSpecialCharsInURL': 2,
    'SpacialCharRatioInURL': 0.09, 'IsHTTPS': 1,
    # Content features = 0 (simulating failed fetch)
    'LineOfCode': 0, 'LargestLineLength': 0, 'HasTitle': 0, 'DomainTitleMatchScore': 0,
    'URLTitleMatchScore': 0, 'HasFavicon': 0, 'Robots': 0, 'IsResponsive': 0,
    'NoOfURLRedirect': 0, 'NoOfSelfRedirect': 0, 'HasDescription': 0, 'NoOfPopup': 0,
    'NoOfiFrame': 0, 'HasExternalFormSubmit': 0, 'HasSocialNet': 0, 'HasSubmitButton': 0,
    'HasHiddenFields': 0, 'HasPasswordField': 0, 'Bank': 0, 'Pay': 0, 'Crypto': 0,
    'HasCopyrightInfo': 0, 'NoOfImage': 0, 'NoOfCSS': 0, 'NoOfJS': 0,
    'NoOfSelfRef': 0, 'NoOfEmptyRef': 0, 'NoOfExternalRef': 0
}
vec2 = np.array([[google_vec.get(f, 0) for f in feat_names]])
p2 = model.predict_proba(vec2)[0]
print('== Google URL features + content zeros ==')
print('  predict_proba:', p2)
print('  P(phishing):', p2[phishing_idx])
print()

# Test 3: URL-only features + realistic content for google.com
google_full = google_vec.copy()
google_full.update({
    'LineOfCode': 800, 'LargestLineLength': 1500, 'HasTitle': 1, 'DomainTitleMatchScore': 85,
    'URLTitleMatchScore': 70, 'HasFavicon': 1, 'Robots': 1, 'IsResponsive': 1,
    'NoOfURLRedirect': 1, 'NoOfSelfRedirect': 0, 'HasDescription': 1, 'NoOfPopup': 0,
    'NoOfiFrame': 0, 'HasExternalFormSubmit': 0, 'HasSocialNet': 1, 'HasSubmitButton': 0,
    'HasHiddenFields': 0, 'HasPasswordField': 0, 'Bank': 0, 'Pay': 0, 'Crypto': 0,
    'HasCopyrightInfo': 1, 'NoOfImage': 30, 'NoOfCSS': 10, 'NoOfJS': 25,
    'NoOfSelfRef': 50, 'NoOfEmptyRef': 5, 'NoOfExternalRef': 15
})
vec3 = np.array([[google_full.get(f, 0) for f in feat_names]])
p3 = model.predict_proba(vec3)[0]
print('== Google URL + realistic content ==')
print('  predict_proba:', p3)
print('  P(phishing):', p3[phishing_idx])
print()

# Now show the distribution of the model's feature importances
fi = model.feature_importances_
top_feats = sorted(zip(feat_names, fi), key=lambda x: -x[1])[:15]
print('== Top 15 feature importances ==')
for fn, fi_val in top_feats:
    print(f'  {fn:<35} {fi_val:.4f}')
