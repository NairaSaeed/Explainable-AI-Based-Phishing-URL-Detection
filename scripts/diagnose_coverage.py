import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from src.feature_extraction import extract_full_features, CONTENT_FEATURES, MIN_CONTENT_COVERAGE

urls = [
    'https://edas.info',
    'https://tudasoge.z27.web.core.windows.net/',
    'https://guide-ledgrr-live.pages.dev/',
]

for url in urls:
    df, meta = extract_full_features(url, fetch_timeout=8)
    row = df.iloc[0].to_dict()
    non_zero_content = [(f, row.get(f,0)) for f in CONTENT_FEATURES if row.get(f,0) != 0]
    zero_content = [f for f in CONTENT_FEATURES if row.get(f,0) == 0]
    print(f"URL: {url}")
    print(f"  fetch_status:     {meta['fetch_status']}")
    print(f"  content_coverage: {meta['content_coverage']}")
    print(f"  is_full_extract:  {meta['is_full_extract']}")
    print(f"  non-zero content ({len(non_zero_content)}): {non_zero_content}")
    print(f"  zero content ({len(zero_content)}): {zero_content}")
    print()
