import csv
import os
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

links_requested = set()
csv_writer = None
csvfile_handle = None

# Define extensions to explicitly skip (add more as needed)
SKIPPED_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".xml",
    ".rss",
    ".atom",  # Styles, scripts, data
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",  # Images
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",  # Videos
    ".mp3",
    ".wav",
    ".ogg",  # Audio
    ".zip",
    ".rar",
    ".tar",
    ".gz",  # Archives
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",  # Fonts
}


# Function to save content to a file
def save_content(url, content, folder):
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")
    filename = os.path.basename(path)

    # Default to index.html if path is empty or a directory
    if not path or path.endswith("/"):
        path = os.path.join(path, "index.html")
        filename = "index.html"

    # Check if the original URL (passed to this function) indicates a PDF
    is_pdf = parsed_url.path.lower().endswith(".pdf")

    # Ensure correct extension
    if is_pdf:
        # Force .pdf extension if the URL indicates it's a PDF
        if not filename.lower().endswith(".pdf"):
            path += ".pdf"  # Append if missing (e.g., redirect changed URL)
    elif "." not in filename:
        # If no extension and not a PDF, assume HTML directory index
        path = os.path.join(path, "index.html")
    elif not filename.lower().endswith((".html", ".htm")):
        # If it has an unknown extension but wasn't filtered earlier
        # and isn't PDF, maybe default to .html? Or log a warning?
        # For now, let's keep its extension but this might be refined.
        pass  # Keep original extension if it's not html/htm/pdf

    file_path = os.path.join(folder, path.lstrip("/"))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.isfile(file_path):
        print(f"File to save: {file_path}")
        try:
            # PDFs need binary write mode
            mode = "wb"  # Use binary mode for all saving, safer for encoding
            with open(file_path, mode) as file:
                # Content from Selenium page_source is str, needs encoding
                # If we fetched PDF directly, content might be bytes already
                if isinstance(content, str):
                    file.write(content.encode("utf-8"))
                else:
                    file.write(content)  # Assume bytes if not string
            print(f"Saved: {file_path}")
        except OSError as e:
            print(f"Error saving file {file_path}: {e}")
    else:
        print(f"File exists already: {file_path}")


# Function to download and save a resource
def download_resource(url, folder, driver):
    global csv_writer, csvfile_handle
    is_pdf = urlparse(url).path.lower().endswith(".pdf")
    content = None
    final_url = url  # Default final_url to initial url

    try:
        print(f"Begin to get URL: {url}")
        # --- For PDFs, Selenium isn't ideal, direct request is better ---
        # --- But staying with Selenium as per original script for now ---
        # --- A better approach for PDFs would use requests library ---
        driver.get(url)
        time.sleep(2)  # Adjust as needed, explicit waits are better
        print(f"End to get URL: {url}")

        final_url = driver.current_url  # Get final URL after potential redirects
        print(f"Final url {final_url}")

        # --- Logging & Flushing (Only for HTML/PDF attempts) ---
        if csv_writer and csvfile_handle:
            csv_writer.writerow([url, final_url])
            csvfile_handle.flush()
        # -------------------------

        if is_pdf:
            # Selenium doesn't reliably get PDF content via page_source.
            # This part is problematic with Selenium for actual PDF download.
            # For demonstration, we'll save an empty file or potentially
            # the source of the PDF viewer page if one exists.
            # A robust solution would use requests for PDF URLs.
            print("Attempting to save PDF link destination.")
            # We will save an empty file as Selenium can't get PDF bytes directly
            # save_content will create the file path correctly.
            content = b""  # Placeholder for PDF content
        else:
            # Get HTML content
            content = driver.page_source  # This is a string

        # Save content using the FINAL URL to determine path/filename
        save_content(final_url, content, folder)
        return content  # Return content for parsing (if HTML)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        # Log failure if desired
        # if csv_writer and csvfile_handle:
        #     csv_writer.writerow([url, f"Failed: {e}"])
        #     csvfile_handle.flush()
        return None


# Main function to crawl the webpage
def crawl_webpage(base_url, folder, driver):
    # --- Filtering Logic ---
    parsed_url = urlparse(base_url)
    path = parsed_url.path.lower()
    filename = os.path.basename(path)

    # 1. Check domain (already done, but good practice)
    if not parsed_url.netloc == urlparse(list_pages["particuliers"]).netloc:
        print(f"Skipping external domain: {base_url}")
        return

    # 2. Check for explicitly skipped extensions
    if any(path.endswith(ext) for ext in SKIPPED_EXTENSIONS):
        print(f"Skipping URL type (by extension): {base_url}")
        return

    # 3. Determine if it's likely HTML or PDF (allow if PDF, HTML, or no extension)
    is_pdf = path.endswith(".pdf")
    is_html = path.endswith((".html", ".htm"))
    has_extension = "." in filename

    if not (is_pdf or is_html or not has_extension):
        print(f"Skipping URL type (unknown/disallowed extension): {base_url}")
        return
    # --- End Filtering Logic ---

    if base_url in links_requested:
        print(f"Already processed or queued: {base_url}")
        return

    print(f"Processing URL: {base_url}")
    links_requested.add(base_url)
    os.makedirs(folder, exist_ok=True)

    # Download resource (will now only be called for HTML/PDF)
    # Note: Selenium isn't great for downloading PDFs directly.
    content = download_resource(base_url, folder, driver)
    if not content:
        return  # Stop if download failed

    # --- Skip parsing for PDFs ---
    if is_pdf:
        print(f"Skipping parsing for PDF: {base_url}")
        return
    # Also skip parsing if content wasn't HTML (e.g., empty PDF content)
    if not isinstance(content, str) or not content.strip():
        print(f"Skipping parsing, no valid HTML content retrieved for: {base_url}")
        return
    # --- End Skip Parsing ---

    try:
        # Proceed with parsing only if it was HTML content
        soup = BeautifulSoup(content, "lxml")
    except Exception as e:
        # Catch potential errors if content is not valid HTML despite checks
        print(f"Failed to parse presumed HTML {base_url}: {e}")
        return

    print(f"########\nParsing links from: {base_url}")
    resources_to_crawl = set()
    for tag in soup.find_all(
        ["a", "link", "script"]
    ):  # Keep finding all links initially
        url = None
        if tag.name == "a" and tag.get("href"):
            url = tag.get("href")
        # We don't need CSS/JS links anymore, but finding them doesn't hurt
        # The filtering at the start of crawl_webpage will skip them anyway
        # elif tag.name == 'link' and tag.get('rel') == ['stylesheet'] and tag.get('href'):
        #     url = tag.get('href')
        # elif tag.name == 'script' and tag.get('src'):
        #      url = tag.get('src')

        if url:
            parsed_link_url = urlparse(url)
            # Skip non-http links like mailto:, tel:, javascript:
            if parsed_link_url.scheme not in ("", "http", "https"):
                continue
            cleaned_url = parsed_link_url._replace(fragment="").geturl()
            full_url = urljoin(base_url, cleaned_url)
            # Check domain again before adding to crawl list
            if urlparse(full_url).netloc == parsed_url.netloc:
                resources_to_crawl.add(full_url)

    print(
        f"Found {len(resources_to_crawl)} potential HTTP links to consider from {base_url}"
    )

    for resource in resources_to_crawl:
        # The recursive call will handle filtering the resource type
        crawl_webpage(resource, folder, driver)


if __name__ == "__main__":
    list_pages = {
        "particuliers": "https://www.bgl.lu/fr/particuliers.html",
    }

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    )

    # --- IMPORTANT PDF HANDLING NOTE ---
    # To properly DOWNLOAD PDFs, Selenium is not the right tool.
    # You'd typically identify PDF links in BeautifulSoup, then use
    # a library like `requests` to download them directly.
    # This script *identifies* PDF links and logs them, but the actual
    # download via Selenium won't save the PDF content correctly.
    # It saves an empty file as a placeholder.
    # ----------------------------------

    driver_service = Service()
    driver = None
    csvfile_handle = None
    log_filename = "url_log.csv"

    try:
        driver = webdriver.Chrome(service=driver_service, options=chrome_options)
        file_exists = os.path.isfile(log_filename)

        with open(log_filename, "a", newline="", encoding="utf-8") as csvfile:
            csvfile_handle = csvfile
            csv_writer = csv.writer(csvfile_handle)

            if not file_exists or os.path.getsize(log_filename) == 0:
                csv_writer.writerow(["Initial URL", "Final URL"])
                csvfile_handle.flush()

            for output_topic, base_url in list_pages.items():
                output_folder = os.path.join("bgl_bnp_resources", output_topic)
                print(f"\n--- Starting crawl for '{output_topic}' ---")
                crawl_webpage(base_url, output_folder, driver)

    except Exception as e:
        print(f"An error occurred during script execution: {e}")
    finally:
        if driver:
            driver.quit()
        print("Script finished.")  # CSV file is closed automatically by 'with'
