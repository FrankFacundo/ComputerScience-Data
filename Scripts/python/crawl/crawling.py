import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# Function to save content to a file
def save_content(url, content, folder):
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")
    if not path:
        path = "index.html"
    file_path = os.path.join(folder, path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        file.write(content)
    print(f"Saved: {file_path}")


# Function to download and save a resource
def download_resource(url, folder):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            save_content(url, response.content, folder)
    except Exception as e:
        print(f"Failed to download {url}: {e}")


# Main function to crawl the webpage
def crawl_webpage(base_url, folder):
    os.makedirs(folder, exist_ok=True)
    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return

    save_content(base_url, response.content, folder)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all resource links
    resources = set()
    for tag in soup.find_all(["a", "link", "script", "img"]):
        if tag.name == "a":
            url = tag.get("href")
        elif tag.name == "link":
            url = tag.get("href")
        elif tag.name == "script":
            url = tag.get("src")
        elif tag.name == "img":
            url = tag.get("src")
        else:
            continue
        if url:
            full_url = urljoin(base_url, url)
            resources.add(full_url)

    for resource in resources:
        download_resource(resource, folder)


if __name__ == "__main__":
    base_url = "https://www.bgl.lu/fr/particuliers.html"
    output_folder = "bgl_bnp_resources"
    crawl_webpage(base_url, output_folder)
