import csv
import os
import warnings
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def save_content(url, content, folder):
    """Saves the content of a URL to a file."""
    try:
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
    except Exception as e:
        print(f"Error saving content for {url}: {e}")


def download_resource(url, folder):
    """Downloads a resource and returns the response, initial URL, and final URL."""
    retries = 5
    try:
        normalized_url = url.encode("latin1").decode("utf-8")
        sanitized_url = unquote(normalized_url).replace(" ", "%20")
    except UnicodeDecodeError:
        print(f"Failed to decode URL using latin1 -> utf-8. Using original URL: {url}")
        sanitized_url = url

    for attempt in range(retries):
        try:
            # response = requests.get(sanitized_url, timeout=10, verify=False)
            response = requests.get(sanitized_url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            final_url = response.url
            print(f"Redirected from {sanitized_url} to final URL: {final_url}")

            # save_content(final_url, response.content, folder)
            return response, sanitized_url, final_url
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to download {sanitized_url}: {e}")

    print(f"Failed to download {sanitized_url} after {retries} attempts.")
    return None, sanitized_url, None


def crawl_webpage(
    base_url,
    folder,
    dev_mode,
    links_requested=None,
    download_count=None,
    url_mappings=None,
):
    """Crawls a webpage, downloads resources, and follows links."""
    if links_requested is None:
        links_requested = set()
    if download_count is None:
        download_count = {"count": 0}
    if url_mappings is None:
        url_mappings = []

    limit_dev_document_update_ingest = 3
    limit = limit_dev_document_update_ingest if dev_mode else None

    if not base_url.startswith("https://www.bgl.lu/fr"):
        return

    os.makedirs(folder, exist_ok=True)

    if base_url in links_requested:
        return
    links_requested.add(base_url)

    if limit is not None and download_count["count"] >= limit:
        return

    response, initial_url, final_url = download_resource(base_url, folder)

    if response and final_url:
        url_mappings.append((initial_url, final_url))

    if not response:
        return

    download_count["count"] += 1

    # if not (base_url.endswith(".html") or base_url.endswith(".pdf")):
    if not (base_url.endswith(".html")):
        return

    if base_url.endswith(".html"):
        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception as e:
            print(f"Failed to parse {base_url}: {e}")
            return

        resources = set()
        for tag in soup.find_all(["a", "link", "script"]):
            url = tag.get("href")

            if url:
                full_url = urljoin(base_url, url)
                # if full_url.endswith(".html") or full_url.endswith(".pdf"):
                if full_url.endswith(".html"):
                    resources.add(full_url)

        for resource in resources:
            if resource not in links_requested:
                crawl_webpage(
                    resource,
                    folder,
                    dev_mode=dev_mode,
                    links_requested=links_requested,
                    download_count=download_count,
                    url_mappings=url_mappings,
                )


def save_url_mappings_to_csv(brute_files_path, url_mappings):
    """Saves the URL mappings to a CSV file."""
    csv_file_path = os.path.join(brute_files_path, "redirect_urls.csv")
    try:
        with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Initial URL", "Final URL"])
            writer.writerows(url_mappings)
        print(f"URL mappings saved to {csv_file_path}")
    except IOError as e:
        print(f"Error writing to CSV file {csv_file_path}: {e}")


def scrap_public_site(brute_files_path: str, dev_mode: bool):
    """Starts the scraping process for the public site."""
    download_count = {"count": 0}
    url_mappings = []
    base_urls = ["https://www.bgl.lu/fr/particuliers.html"]
    for base_url in base_urls:
        crawl_webpage(
            base_url=base_url,
            folder=brute_files_path,
            dev_mode=dev_mode,
            download_count=download_count,
            url_mappings=url_mappings,
        )

    save_url_mappings_to_csv(brute_files_path, url_mappings)


if __name__ == "__main__":
    brute_files_path = (
        "/home/frank/code/ComputerScience-Data/Scripts/python/crawl/files"
    )
    scrap_public_site(brute_files_path=brute_files_path, dev_mode=False)
