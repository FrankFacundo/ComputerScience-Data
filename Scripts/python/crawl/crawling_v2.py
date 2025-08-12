import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def save_content(url, content, folder):
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")
    if not path:
        path = "index.html"
    file_path = os.path.join(folder, path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        with open(file_path, "wb") as file:
            file.write(content)
        print(f"Saved: {url}")
    else:
        print("File exists already.")


def download_resource(url, folder):
    retries = 5
    try:
        normalized_url = url.encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        print(f"Failded to decode URL using latin1 -> utf-8. Using original URL: {url}")
        normalized_url = url

    for attempt in range(retries):
        try:
            response = requests.get(normalized_url, timeout=10, verify=False)
            if response.status_code == 200:
                save_content(normalized_url, response.content, folder)
                return response
        except Exception as e:
            print(f"Attempt {attempt +1} failed to download {normalized_url}: {e}")
    print(f"Failed to download {normalized_url} after {retries} attempts.")


def crawl_webpage(
    base_url,
    folder,
    dev_mode,
    links_requested=None,
    download_count=None,
    follow_links=True,
):
    if links_requested is None:
        links_requested = set()
    if download_count is None:
        download_count = {"count": 0}

    limit_dev_document_update_ingest = 3
    limit = limit_dev_document_update_ingest if dev_mode else None

    if not base_url.startswith("https://www.bgl.lu"):
        return

    os.makedirs(folder, exist_ok=True)

    if base_url in links_requested:
        return
    links_requested.add(base_url)

    if limit is not None and download_count["count"] >= limit:
        return

    response = download_resource(base_url, folder)
    if not response:
        return

    download_count["count"] += 1

    if not (base_url.endswith(".html") or base_url.endswith(".pdf")):
        return

    if base_url.endswith(".html"):
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception as e:
            print(f"Failed to parse {base_url}: {e}")
            return

        resources = set()
        for tag in soup.find_all(["a", "link", "script"]):
            if tag.name in ["a", "link"]:
                url = tag.get("href")
            else:
                continue

            if url:
                full_url = urljoin(base_url, url)
                if full_url.endswith(".html") or full_url.endswith(".pdf"):
                    resources.add(full_url)

        for resource in resources:
            if resource not in links_requested:
                crawl_webpage(
                    resource,
                    folder,
                    dev_mode=dev_mode,
                    links_requested=links_requested,
                    download_count=download_count,
                )


def scrap_public_site(brute_files_path: str, dev_mode: bool):
    download_count = {"count": 0}
    base_urls = ["https://www.bgl.lu/fr/particuliers.html"]
    for base_url in base_urls:
        crawl_webpage(
            base_url=base_url,
            folder=brute_files_path,
            dev_mode=dev_mode,
            download_count=download_count,
            follow_links=False,
        )


if __name__ == "__main__":
    brute_files_path = (
        "/home/frank/code/ComputerScience-Data/Scripts/python/crawl/files"
    )
    scrap_public_site(brute_files_path=brute_files_path, dev_mode=False)
