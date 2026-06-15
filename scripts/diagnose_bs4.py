import sys, requests
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append('.')
from bs4 import BeautifulSoup

urls = ['https://edas.info', 'https://guide-ledgrr-live.pages.dev/']
for url in urls:
    r = requests.get(url, timeout=8, verify=False,
                     headers={'User-Agent':'Mozilla/5.0'},
                     allow_redirects=True)
    soup = BeautifulSoup(r.text, 'html.parser')
    html = r.text.lower()
    title = soup.find('title')
    css_links = soup.find_all('link', rel=lambda r: r and 'stylesheet' in ' '.join(r).lower())
    imgs = soup.find_all('img')
    favicons = soup.find_all('link', rel=lambda r: r and 'icon' in ' '.join(r).lower())
    scripts = soup.find_all('script')
    hrefs = [a.get('href','') for a in soup.find_all(['a','link','area'])]
    domain = url.split('/')[2]
    self_refs = [h for h in hrefs if h.startswith('/') or domain in h]
    ext_refs = [h for h in hrefs if h.startswith('http') and domain not in h]
    print(f"URL: {url}")
    print(f"  Status: {r.status_code}, redirects: {len(r.history)}")
    print(f"  Title: {title}")
    print(f"  CSS links: {len(css_links)}")
    print(f"  Images: {len(imgs)}")
    print(f"  Favicons: {len(favicons)}")
    print(f"  Scripts: {len(scripts)}")
    print(f"  Self-refs: {len(self_refs)}")
    print(f"  Ext-refs: {len(ext_refs)}")
    print(f"  Has viewport: {bool(soup.find('meta', attrs={'name':'viewport'}))}")
    print(f"  Has description: {bool(soup.find('meta', attrs={'name':'description'}))}")
    print(f"  Has submit: {bool(soup.find('input', {'type':'submit'}) or soup.find('button', {'type':'submit'}))}")
    print(f"  Has copyright: {'copyright' in html or chr(169) in r.text.lower()}")
    print(f"  Lines of code: {len(r.text.splitlines())}")
    print()
