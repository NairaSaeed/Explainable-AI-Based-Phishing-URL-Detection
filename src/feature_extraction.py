"""
src/feature_extraction.py
==========================
Paper-aligned feature extraction for the Phishing URL Detection GUI.

Two modes:
  1. Full-feature extraction  – URL lexical + HTML/content (requires page fetch)
     Returns (pd.DataFrame, metadata_dict)
  2. URL-only extraction      – 27 lexical features from URL string alone
     Returns pd.DataFrame

Full-feature set aligns exactly with the PhiUSIIL training columns loaded from
outputs/models/full_feature_list.json.  Content features are extracted via
requests + BeautifulSoup.  On any network / parse failure the function returns
is_full_extract=False so the caller can safely fall back to the URL-only model.

NOTE: Detected signals are purely for analyst display.
      They NEVER override the ML model prediction.
"""

from __future__ import annotations

import re
import math
import warnings
import urllib.parse
from typing import Tuple

import requests
import pandas as pd

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
except Exception:
    pass

warnings.filterwarnings("ignore")

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────
IP_RE   = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
KW_LIST = ['login', 'verify', 'update', 'secure', 'account', 'bank',
           'paypal', 'wallet', 'invoice', 'password', 'signin', 'confirm']

# Content features that must be non-zero for full-feature mode to be trusted
CONTENT_FEATURES = [
    'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
    'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
    'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
    'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
    'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef',
]

# Minimum fraction of content features that must be non-zero to trust full-feature mode.
# Raised slightly from 0.40 because very sparse pages (JS-redirect stubs) should
# not be passed to the full-feature model.
MIN_CONTENT_COVERAGE = 0.30   # at least 30% of content features must be non-zero


# ── Internal helpers ──────────────────────────────────────────────────────────
def _entropy(s: str) -> float:
    s = str(s)
    if not s:
        return 0.0
    cnt: dict[str, int] = {}
    for c in s:
        cnt[c] = cnt.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in cnt.values())


def _safe_parse(url: str) -> urllib.parse.ParseResult:
    try:
        return urllib.parse.urlparse(url if '://' in url else 'https://' + url)
    except Exception:
        return urllib.parse.urlparse('')


def _url_lexical(url: str, parsed: urllib.parse.ParseResult) -> dict:
    """Lexical features matching the PhiUSIIL full-feature column names."""
    domain = parsed.hostname or ''
    parts  = domain.split('.')
    tld    = parts[-1] if len(parts) > 1 else ''
    letters = sum(c.isalpha() for c in url)
    digits  = sum(c.isdigit() for c in url)
    special = sum(not c.isalnum() and c not in ('/', ':', '.', '-', '_') for c in url)
    return {
        'URLLength':                  len(url),
        'DomainLength':               len(domain),
        'IsDomainIP':                 1 if IP_RE.match(domain) else 0,
        'URLSimilarityIndex':         0.0,    # filled after page fetch
        'CharContinuationRate':       0.0,    # filled after page fetch
        'TLDLegitimateProb':          0.5,    # heuristic placeholder
        'URLCharProb':                0.5,    # heuristic placeholder
        'TLDLength':                  len(tld),
        'NoOfSubDomain':              max(0, len(parts) - 2),
        'HasObfuscation':             1 if ('%' in url or '@' in url) else 0,
        'NoOfObfuscatedChar':         url.count('%'),
        'ObfuscationRatio':           url.count('%') / max(1, len(url)),
        'NoOfLettersInURL':           letters,
        'LetterRatioInURL':           letters / max(1, len(url)),
        'NoOfDegitsInURL':            digits,
        'DegitRatioInURL':            digits / max(1, len(url)),
        'NoOfEqualsInURL':            url.count('='),
        'NoOfQMarkInURL':             url.count('?'),
        'NoOfAmpersandInURL':         url.count('&'),
        'NoOfOtherSpecialCharsInURL': special,
        'SpacialCharRatioInURL':      special / max(1, len(url)),
        'IsHTTPS':                    1 if url.lower().startswith('https') else 0,
    }


def _extract_content_features(
    url: str,
    parsed: urllib.parse.ParseResult,
    fetch_timeout: int = 6,
) -> Tuple[dict, str, int]:
    """
    Fetch the page and extract content features.
    Returns (features_dict, fetch_status, num_redirects).
    fetch_status: 'success' | 'blocked' | 'timeout' | 'ssl_error' |
                  'connection_error' | 'error:<type>' | 'no_bs4'
    On any failure, returns a dict of zeros so callers can check fetch_status.
    """
    zero = {f: 0 for f in CONTENT_FEATURES}
    zero.update({'URLSimilarityIndex': 0.0, 'CharContinuationRate': 0.0})

    if not _BS4_OK:
        return zero, 'no_bs4 (install beautifulsoup4)', 0

    domain = parsed.hostname or ''
    try:
        resp = requests.get(
            url,
            timeout=fetch_timeout,
            verify=False,
            allow_redirects=True,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/124.0.0.0 Safari/537.36'
                )
            },
        )
        # 4xx/5xx responses – treat as blocked/failed
        if resp.status_code in (403, 404, 429, 500, 503):
            return zero, f'blocked_{resp.status_code}', len(resp.history)

        n_redirects = len(resp.history)
        html_raw    = resp.text
        soup        = BeautifulSoup(html_raw, 'html.parser')
        html        = html_raw.lower()

    except requests.exceptions.Timeout:
        return zero, 'timeout', 0
    except requests.exceptions.SSLError:
        return zero, 'ssl_error', 0
    except requests.exceptions.ConnectionError:
        return zero, 'connection_error', 0
    except Exception as exc:
        return zero, f'error:{type(exc).__name__}', 0

    # ── Parse features ────────────────────────────────────────────────
    lines    = html_raw.splitlines()
    loc      = len(lines)
    max_line = max((len(l) for l in lines), default=0)

    title_tag  = soup.find('title')
    title_text = title_tag.get_text(strip=True).lower() if title_tag else ''

    # DomainTitleMatchScore: % of domain tokens found in title
    domain_tokens = [t for t in re.split(r'[.\-_]', domain.lower()) if len(t) > 2]
    dom_match = (
        sum(1 for t in domain_tokens if t in title_text) / len(domain_tokens)
        if domain_tokens else 0.0
    )
    dom_match_score = round(dom_match * 100, 1)

    # URLTitleMatchScore: overlap between URL path tokens and title
    url_tokens = re.findall(r'[a-z]{3,}', parsed.path.lower())
    url_match  = (
        sum(1 for t in url_tokens if t in title_text) / len(url_tokens)
        if url_tokens else 0.0
    )
    url_match_score = round(url_match * 100, 1)

    # URLSimilarityIndex: max of both match scores
    url_sim = max(dom_match_score, url_match_score)

    # CharContinuationRate: density of repeated-char runs in first 5K of html
    runs   = re.findall(r'(.)\1+', html[:5000])
    cc_val = round(len(runs) / max(1, len(html[:5000])), 6)

    # Favicon
    has_favicon = int(bool(
        soup.find('link', rel=lambda r: r and 'icon' in ' '.join(r).lower())
    ))

    # Robots meta
    has_robots = int(bool(
        soup.find('meta', attrs={'name': re.compile(r'robots', re.I)})
    ))

    # Viewport → responsive
    is_responsive = int(bool(
        soup.find('meta', attrs={'name': re.compile(r'viewport', re.I)})
    ))

    # Self-redirects (JS window.location)
    self_redir = html.count('window.location')

    # Meta description
    has_desc = int(bool(
        soup.find('meta', attrs={'name': re.compile(r'description', re.I)})
    ))

    # Popups
    n_popup = html.count('window.open')

    # iFrames
    n_iframe = len(soup.find_all('iframe'))

    # External form submit
    forms    = soup.find_all('form')
    ext_form = int(any(
        f.get('action', '').startswith('http') and domain not in f.get('action', '')
        for f in forms
    ))

    # Social networks
    social_nets = ['facebook.com', 'twitter.com', 'instagram.com',
                   'linkedin.com', 'youtube.com', 'x.com']
    has_social = int(any(s in html for s in social_nets))

    # Submit button
    has_submit = int(
        bool(soup.find('input', {'type': 'submit'})) or
        bool(soup.find('button', {'type': 'submit'})) or
        bool(soup.find('button', string=re.compile(r'submit|login|sign in', re.I)))
    )

    # Hidden fields
    has_hidden = int(len(soup.find_all('input', {'type': 'hidden'})) > 0)

    # Password field
    has_pwd = int(len(soup.find_all('input', {'type': 'password'})) > 0)

    # Keyword flags
    has_bank   = int('bank' in html)
    has_pay    = int('pay' in html)
    has_crypto = int(any(k in html for k in ['crypto', 'bitcoin', 'ethereum', 'blockchain']))
    has_copy   = int('copyright' in html or '\u00a9' in html or '&copy' in html)

    # Counts
    n_img = len(soup.find_all('img'))
    n_css = len(soup.find_all('link', rel=lambda r: r and 'stylesheet' in ' '.join(r).lower()))
    n_js  = len(soup.find_all('script'))

    # Reference counts
    all_hrefs = [a.get('href', '') for a in soup.find_all(['a', 'link', 'area'])]
    self_ref  = sum(1 for h in all_hrefs if h.startswith('/') or (domain and domain in h))
    empty_ref = sum(1 for h in all_hrefs if h.strip() in ('#', '', 'javascript:void(0)', 'javascript:;'))
    ext_ref   = sum(1 for h in all_hrefs if h.startswith('http') and domain and domain not in h)

    feats = {
        'URLSimilarityIndex':    url_sim,
        'CharContinuationRate':  cc_val,
        'LineOfCode':            loc,
        'LargestLineLength':     max_line,
        'HasTitle':              int(title_tag is not None),
        'DomainTitleMatchScore': dom_match_score,
        'URLTitleMatchScore':    url_match_score,
        'HasFavicon':            has_favicon,
        'Robots':                has_robots,
        'IsResponsive':          is_responsive,
        'NoOfURLRedirect':       n_redirects,
        'NoOfSelfRedirect':      self_redir,
        'HasDescription':        has_desc,
        'NoOfPopup':             n_popup,
        'NoOfiFrame':            n_iframe,
        'HasExternalFormSubmit': ext_form,
        'HasSocialNet':          has_social,
        'HasSubmitButton':       has_submit,
        'HasHiddenFields':       has_hidden,
        'HasPasswordField':      has_pwd,
        'Bank':                  has_bank,
        'Pay':                   has_pay,
        'Crypto':                has_crypto,
        'HasCopyrightInfo':      has_copy,
        'NoOfImage':             n_img,
        'NoOfCSS':               n_css,
        'NoOfJS':                n_js,
        'NoOfSelfRef':           self_ref,
        'NoOfEmptyRef':          empty_ref,
        'NoOfExternalRef':       ext_ref,
    }
    return feats, 'success', n_redirects


# ── Public API ────────────────────────────────────────────────────────────────

def extract_full_features(url: str, fetch_timeout: int = 6) -> tuple[pd.DataFrame, dict]:
    """
    Attempt full PhiUSIIL-aligned feature extraction.

    Returns
    -------
    df   : pd.DataFrame with one row, columns = all extracted feature names
    meta : dict with keys:
             fetch_status      – 'success' | 'timeout' | 'blocked_403' | …
             is_full_extract   – True only when page was fetched AND content
                                 coverage >= MIN_CONTENT_COVERAGE
             content_coverage  – fraction of CONTENT_FEATURES that are non-zero
    """
    parsed  = _safe_parse(url)
    lex     = _url_lexical(url, parsed)
    content, fetch_status, _ = _extract_content_features(url, parsed, fetch_timeout)

    # Merge: content values override lex placeholders where they exist
    feats = {**lex, **content}

    # Evaluate coverage
    non_zero = sum(1 for f in CONTENT_FEATURES if feats.get(f, 0) != 0)
    coverage = non_zero / len(CONTENT_FEATURES)
    is_full  = (fetch_status == 'success') and (coverage >= MIN_CONTENT_COVERAGE)

    df = pd.DataFrame([feats])
    meta = {
        'fetch_status':     fetch_status,
        'is_full_extract':  is_full,
        'content_coverage': round(coverage, 3),
    }
    return df, meta


def extract_url_only_features(url: str) -> pd.DataFrame:
    """
    Extract 27 lexical URL-only features (matches notebook Experiment 2).
    Never raises – returns zeros on any parse error.
    """
    url = str(url).strip()
    try:
        p = _safe_parse(url)
    except Exception:
        p = urllib.parse.urlparse('')

    host  = p.hostname or ''
    path  = p.path or ''
    query = p.query or ''
    low   = url.lower()

    d: dict = {
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
        'has_ip_address':            1 if IP_RE.match(host) else 0,
        'has_https':                 int(url.startswith('https')),
        'has_at_symbol':             int('@' in url),
        'has_double_slash_redirect': int('//' in path),
        'entropy':                   round(_entropy(url), 4),
    }
    for kw in KW_LIST:
        d[f'count_{kw}'] = low.count(kw)
    return pd.DataFrame([d])


def extract_all_features_from_url(url: str, fetch_timeout: int = 6) -> dict:
    """Legacy shim – returns flat feature dict (full-feature attempt)."""
    df, _ = extract_full_features(url, fetch_timeout)
    return df.iloc[0].to_dict()


def get_detected_signals(feats: dict, mode: str) -> list[str]:
    """
    Return a list of human-readable analyst signals for display only.
    These NEVER override the ML model prediction.
    """
    signals = []
    if mode == 'url_only':
        host = feats.get('hostname_length', 0)
        if feats.get('has_ip_address', 0):
            signals.append('IP address used as hostname')
        if not feats.get('has_https', 1):
            signals.append('No HTTPS')
        if feats.get('url_length', 0) > 75:
            signals.append(f"Long URL ({feats['url_length']} chars)")
        if feats.get('num_subdomains', 0) >= 3:
            signals.append(f"Many subdomains ({feats['num_subdomains']})")
        if feats.get('has_at_symbol', 0):
            signals.append('@ symbol in URL')
        if feats.get('entropy', 0) > 4.5:
            signals.append(f"High URL entropy ({feats['entropy']:.2f})")
        if feats.get('num_hyphens', 0) >= 4:
            signals.append(f"Many hyphens ({feats['num_hyphens']})")
        sus_kws = [kw for kw in KW_LIST if feats.get(f'count_{kw}', 0) > 0]
        if sus_kws:
            signals.append(f"Suspicious keywords: {', '.join(sus_kws)}")
    else:
        if feats.get('IsDomainIP', 0):
            signals.append('IP address used as hostname')
        if not feats.get('IsHTTPS', 1):
            signals.append('No HTTPS')
        if feats.get('URLLength', 0) > 75:
            signals.append(f"Long URL ({feats['URLLength']} chars)")
        if feats.get('NoOfSubDomain', 0) >= 3:
            signals.append(f"Many subdomains ({feats['NoOfSubDomain']})")
        if feats.get('HasObfuscation', 0):
            signals.append('URL obfuscation detected')
        if feats.get('HasPasswordField', 0):
            signals.append('Password field detected in page')
        if feats.get('HasHiddenFields', 0):
            signals.append('Hidden form fields detected')
        if feats.get('HasExternalFormSubmit', 0):
            signals.append('Form submits to external domain')
        if feats.get('NoOfPopup', 0) > 0:
            signals.append(f"Popup windows ({feats['NoOfPopup']})")
        if feats.get('NoOfiFrame', 0) > 2:
            signals.append(f"Multiple iFrames ({feats['NoOfiFrame']})")
        if feats.get('Bank', 0) or feats.get('Pay', 0):
            signals.append('Financial keywords in page content')
    return signals
