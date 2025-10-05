# ... existing imports ...
import xml.etree.ElementTree as ET  #  NEW

import requests


# ---------- 1.  S I T E M A P   H E L P E R ---------------------------------
def get_sitemap_urls(domain: str):
    """
    Collect every <loc> entry declared in the site’s sitemap(s).
    """
    robots_url = f"https://{domain}/robots.txt"
    sitemap_candidates = []

    # Robots.txt may list several sitemap locations
    try:
        r = requests.get(robots_url, timeout=10, verify=False)
        sitemap_candidates = [
            line.split(":", 1)[1].strip()
            for line in r.text.splitlines()
            if line.lower().startswith("sitemap:")
        ]
    except Exception:
        pass

    if not sitemap_candidates:  # fallback
        sitemap_candidates = [f"https://{domain}/sitemap.xml"]

    urls: set[str] = set()
    for sm_url in sitemap_candidates:
        try:
            r = requests.get(sm_url, timeout=10, verify=False)
            tree = ET.fromstring(r.content)
            loc_tag = "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
            for loc in tree.iter(loc_tag):
                urls.add(loc.text.strip())
        except Exception as e:
            print(f"Could not parse sitemap {sm_url}: {e}")
    return urls


base_urls = get_sitemap_urls("www.bgl.lu") | {"https://www.bgl.lu"}
print(base_urls)
