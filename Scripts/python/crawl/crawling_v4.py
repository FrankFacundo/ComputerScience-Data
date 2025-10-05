import os
import re
import warnings
from urllib.parse import unquote, urljoin, urlparse

import matplotlib.pyplot as plt  #  NEW
import networkx as nx  #  NEW
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)


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
        sanitized_url = unquote(normalized_url).replace(" ", "%20")
    except UnicodeDecodeError:
        print(f"Failded to decode URL using latin1 -> utf-8. Using original URL: {url}")
        normalized_url = url

    for attempt in range(retries):
        try:
            response = requests.get(sanitized_url, timeout=10, verify=False)
            if response.status_code == 200:
                if response.status_code == 200:
                    print(
                        f"Redirected from {sanitized_url} to final URL: {response.url}"
                    )
                    final_url = response.url
                else:
                    final_url = response.url
                save_content(final_url, response.content, folder)
                return response
        except Exception as e:
            print(f"Attempt {attempt +1} failed to download {sanitized_url}: {e}")
    print(f"Failed to download {sanitized_url} after {retries} attempts.")


def crawl_webpage(
    base_url,
    folder,
    dev_mode,
    graph,
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

    if not (
        (base_url.startswith("https://www.bgl.lu"))
        or (base_url.startswith("www.bgl.lu"))
        or (base_url.startswith("bgl.lu"))
    ):
        return

    os.makedirs(folder, exist_ok=True)

    if base_url in links_requested:
        return
    links_requested.add(base_url)
    graph.add_node(base_url)
    if limit is not None and download_count["count"] >= limit:
        return

    response = download_resource(base_url, folder)
    if not response:
        return

    download_count["count"] += 1

    if not (base_url.endswith(".html") or base_url.endswith(".pdf")):
        # if not (base_url.endswith(".html")):
        return
        # if not re.search(r"\.html", base_url):
        # return
    if re.search(r"\.html", base_url):
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
                    # if re.search(r"\.html", full_url):
                    resources.add(full_url)
                    graph.add_edge(base_url, full_url)  #  NEW ─ add link to graph

        for resource in resources:
            if resource not in links_requested:
                crawl_webpage(
                    resource,
                    folder,
                    dev_mode=dev_mode,
                    graph=graph,
                    links_requested=links_requested,
                    download_count=download_count,
                )


def scrap_public_site(brute_files_path: str, dev_mode: bool):
    download_count = {"count": 0}
    graph = nx.DiGraph()
    links_requested: set[str] = set()  # ← NEW ─ shared across base_urls

    # base_urls = ["https://www.bgl.lu/fr/particuliers.html"]
    # base_urls = ["https://www.bgl.lu/fr/entreprises/startups.html"]
    base_urls = [
        "https://www.bgl.lu/fr/particuliers.html",
        "https://www.bgl.lu/fr/entreprises/startups.html",
        "https://www.bgl.lu/fr/entreprises/bgl-bnp-paribas-development.html",
        "https://www.bgl.lu/fr/welcomeing",
        # … add more …
    ]
    for base_url in base_urls:
        crawl_webpage(
            base_url=base_url,
            folder=brute_files_path,
            dev_mode=dev_mode,
            graph=graph,
            links_requested=links_requested,  # ← pass shared set
            download_count=download_count,
            follow_links=False,
        )
    # -------- SAVE + PRINT GRAPH --------------------------------------- #
    graph_path = os.path.join(brute_files_path, "web_graph.gml")
    nx.write_gml(graph, graph_path)
    print(f"\nGraph saved to: {graph_path}")
    print(f"Total nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    print("Edges:")
    # for src, dst in graph.edges():
    #     print(f"  {src}  -->  {dst}")

    # Optional: visualise and save as PNG
    img_path = os.path.join(brute_files_path, "web_graph.png")
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(graph, node_size=60, font_size=5, arrowsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    print(f"Graph image saved to: {img_path}")


if __name__ == "__main__":
    brute_files_path = (
        "/home/frank/code/ComputerScience-Data/Scripts/python/crawl/files"
    )
    scrap_public_site(brute_files_path=brute_files_path, dev_mode=False)
