import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

links_requested = set()


# Function to save content to a file
def save_content(url, content, folder):
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")
    if not path:
        path = "index.html"
    file_path = os.path.join(folder, path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        print(f"File to save: {file_path}")
        with open(file_path, "wb") as file:
            file.write(content)
        print(f"Saved: {file_path}")
    else:
        print("File exists already.")


# Function to download and save a resource
def download_resource(url, folder):
    try:
        print(f"Begin to get URL: {url}")

        response = requests.get(url, timeout=5)
        print(f"End to get URL: {url}")
        if response.status_code == 200:
            save_content(url, response.content, folder)
            return response
        return
    except Exception as e:
        print(f"Failed to download {url}: {e}")


# Main function to crawl the webpage
def crawl_webpage(base_url, folder):
    if not base_url.startswith("https://www.bgl.lu/"):
        return
    os.makedirs(folder, exist_ok=True)

    links_requested.add(base_url)

    response = download_resource(base_url, folder)
    if not response:
        return

    if base_url.endswith("pdf"):
        return

    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception as e:
        print(f"Failed to parse {base_url}: {e}")

    print("########")
    print("base_url : ", base_url)
    # Find all resource links
    resources = set()
    for tag in soup.find_all(["a", "link", "script"]):
        if tag.name == "a":
            url = tag.get("href")
        elif tag.name == "link":
            url = tag.get("href")
        else:
            continue

        if url:
            full_url = urljoin(base_url, url)
            resources.add(full_url)

    print(f"Crawling following resources: {resources}")
    print("base_url : ", base_url)
    for resource in resources:
        if resource not in links_requested:
            crawl_webpage(resource, folder)


if __name__ == "__main__":
    list_pages = {
        "particuliers": "https://www.bgl.lu/fr/particuliers.html",
    }

    for output_folders, base_url in list_pages.items():
        crawl_webpage(base_url, "bgl_bnp_resources")
