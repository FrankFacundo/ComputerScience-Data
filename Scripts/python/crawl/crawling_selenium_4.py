import csv
import os
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# --- Imports for Explicit Waits ---
from selenium.webdriver.support.ui import WebDriverWait

# ---------------------------------

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
    # is_pdf = parsed_url.path.lower().endswith(".pdf")

    # Ensure correct extension
    if False:
        # Force .pdf extension if the URL indicates it's a PDF
        if not filename.lower().endswith(".pdf"):
            # Handle cases where redirect might change URL but content is PDF
            # Ensure the path ends with .pdf if it was intended to be one
            if "." in filename:  # Replace existing extension if needed
                path = os.path.splitext(path)[0] + ".pdf"
            else:  # Append if no extension
                path += ".pdf"
    elif "." not in filename:
        # If no extension and not a PDF, assume HTML directory index
        path = os.path.join(path, "index.html")
    elif not filename.lower().endswith((".html", ".htm")):
        # Keep original extension if it's not html/htm/pdf
        pass

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
        except Exception as e:  # Catch potential encoding errors too
            print(f"Unexpected error saving file {file_path}: {e}")
    else:
        print(f"File exists already: {file_path}")


# ────────────────────────────────────────────────────────────────────────────────
# Helper: wait until the browser is "network‑idle"
# ────────────────────────────────────────────────────────────────────────────────
def _wait_for_network_idle(driver, total_timeout=20, idle_secs=2, poll=0.4):
    """
    Return when the number of network requests reported by the Performance API
    stays unchanged for `idle_secs` seconds, or raise TimeoutException.
    Works with Chromium‑based drivers (Chrome / Edge). If you use GeckoDriver,
    fall back to a simple `time.sleep(idle_secs)`.
    """
    try:
        prev_res_count = driver.execute_script(
            "return performance.getEntriesByType('resource').length"
        )
    except Exception:
        # Performance API not available (e.g. Firefox) – fall back
        time.sleep(idle_secs)
        return

    last_change = time.time()
    deadline = time.time() + total_timeout

    while time.time() < deadline:
        time.sleep(poll)
        cur_res_count = driver.execute_script(
            "return performance.getEntriesByType('resource').length"
        )
        if cur_res_count != prev_res_count:  # network activity seen
            prev_res_count = cur_res_count
            last_change = time.time()
        elif time.time() - last_change >= idle_secs:  # stayed quiet long enough
            return
    raise TimeoutException("Timed out waiting for network idle")


# ────────────────────────────────────────────────────────────────────────────────
# Main routine
# ────────────────────────────────────────────────────────────────────────────────
def download_resource(url, folder, driver, wait_timeout=15):
    """
    Loads `url`, waits for DOM ready *and* for JS‑driven network activity to calm
    down, then saves the **rendered** HTML (or a zero‑byte placeholder for PDFs).
    Returns the content that got written to disk, or None on failure.
    """
    global csv_writer, csvfile_handle

    print(f"Begin to get URL: {url}")
    try:
        driver.get(url)

        # 1)  document.readyState === 'complete'
        WebDriverWait(driver, wait_timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        # 2)  now wait until dynamic scripts have done their work
        _wait_for_network_idle(driver, total_timeout=wait_timeout)

    except TimeoutException:
        print(f"Timeout while loading / waiting for JS on: {url}")
        if csv_writer and csvfile_handle:
            csv_writer.writerow([url, "Failed: Page Load or Network‑Idle Timeout"])
            csvfile_handle.flush()
        return None
    except Exception as e:
        print(f"Unexpected error getting {url}: {e}")
        if csv_writer and csvfile_handle:
            csv_writer.writerow([url, f"Failed: {e}"])
            csvfile_handle.flush()
        return None

    # ── At this point the page is as “settled” as we can reasonably make it ──
    final_url = driver.current_url
    # is_pdf = urlparse(final_url).path.lower().endswith(".pdf")

    # Tentative success log
    if csv_writer and csvfile_handle:
        csv_writer.writerow([url, final_url])
        csvfile_handle.flush()

    try:
        if False:
            print(f"Identified PDF → placeholder only: {final_url}")
            content = b""
            save_content(final_url, content, folder)
            return content

        # Fully rendered DOM (including anything injected by your Adobe / Medallia scripts)
        content = driver.execute_script("return document.documentElement.outerHTML")

        # Optional: simple custom‑404 sniff
        if "<title>" in content.lower() and "file not found" in content.lower():
            print(f"Custom 404 detected on: {final_url}")
            if csv_writer and csvfile_handle:
                csv_writer.writerow(
                    [url, f"Failed: Custom 404 (Final URL: {final_url})"]
                )
                csvfile_handle.flush()
            return None

        save_content(final_url, content, folder)
        return content

    except Exception as e:
        print(f"Error while saving / analysing {final_url}: {e}")
        if csv_writer and csvfile_handle:
            csv_writer.writerow([url, f"Failed: {e}"])
            csvfile_handle.flush()
        return None


# Main function to crawl the webpage
def crawl_webpage(base_url, folder, driver):
    # --- Filtering Logic ---
    parsed_base_url = urlparse(base_url)  # Parse the URL being processed
    path = parsed_base_url.path.lower()
    filename = os.path.basename(path)

    # 1. Check domain (ensure we stay within the target site)
    # Use the netloc from the *initial* list_pages URL for comparison
    initial_domain = urlparse(
        list(list_pages.values())[0]
    ).netloc  # Assumes single domain target
    if parsed_base_url.netloc != initial_domain:
        print(f"Skipping external domain: {base_url}")
        return

    # 2. Check for explicitly skipped extensions
    if any(path.endswith(ext) for ext in SKIPPED_EXTENSIONS):
        print(f"Skipping URL type (by extension): {base_url}")
        return

    # 3. Determine if it's likely HTML or PDF (allow if PDF, HTML, or no extension)
    # is_pdf = path.endswith(".pdf")
    is_html = path.endswith((".html", ".htm"))
    has_extension = "." in filename

    # if not (is_pdf or is_html or not has_extension or path == "/" or path == ""):
    if not (is_html or not has_extension or path == "/" or path == ""):
        print(f"Skipping URL type (unknown/disallowed extension): {base_url}")
        return
    if not (path.startswith("/fr")):
        return
    # --- End Filtering Logic ---

    if base_url in links_requested:
        print(f"Already processed or queued: {base_url}")
        return

    print(f"Processing URL: {base_url}")
    links_requested.add(base_url)
    os.makedirs(folder, exist_ok=True)

    # Download resource (will now only be called for potentially valid HTML/PDF)
    content = download_resource(base_url, folder, driver)
    if (
        content is None
    ):  # Check for None explicitly, as empty string/bytes might be valid
        print(
            f"Download failed or skipped for {base_url}, stopping processing for this URL."
        )
        return  # Stop if download failed or returned None

    # --- Skip parsing for PDFs ---
    # Check based on the FINAL URL after download attempt
    final_url_path = urlparse(driver.current_url).path.lower()  # Re-check final URL
    if final_url_path.endswith(".pdf"):
        print(f"Skipping parsing for PDF (final URL): {driver.current_url}")
        return
    # Also skip parsing if content wasn't HTML (e.g., empty PDF placeholder)
    if not isinstance(content, str) or not content.strip():
        print(
            f"Skipping parsing, no valid HTML content retrieved for: {base_url} (Final: {driver.current_url})"
        )
        return
    # --- End Skip Parsing ---

    try:
        # Proceed with parsing only if it was HTML content
        soup = BeautifulSoup(content, "lxml")
    except Exception as e:
        # Catch potential errors if content is not valid HTML despite checks
        print(f"Failed to parse presumed HTML {base_url}: {e}")
        return

    print(f"########\nParsing links from: {base_url} (Final: {driver.current_url})")
    resources_to_crawl = set()
    for tag in soup.find_all(["a"]):  # Primarily interested in 'a' tags for navigation
        href = tag.get("href")
        if href:
            parsed_link_url = urlparse(href)
            # Skip non-http links like mailto:, tel:, javascript:, #fragments
            if parsed_link_url.scheme not in ("", "http", "https") or href.startswith(
                "#"
            ):
                continue

            # Clean fragment and create absolute URL using the *final* URL as base
            cleaned_url = parsed_link_url._replace(fragment="").geturl()
            full_url = urljoin(driver.current_url, cleaned_url)  # Use final URL as base

            # Check domain again before adding to crawl list (compare to initial domain)
            if urlparse(full_url).netloc == initial_domain:
                # Final filter before adding: Ensure it's not a skipped type
                link_path = urlparse(full_url).path.lower()
                link_filename = os.path.basename(link_path)
                # link_is_pdf = link_path.endswith(".pdf")
                link_is_html = link_path.endswith((".html", ".htm"))
                link_has_extension = "." in link_filename

                if any(link_path.endswith(ext) for ext in SKIPPED_EXTENSIONS):
                    # print(f"Skipping discovered link (skipped extension): {full_url}")
                    continue

                if not (
                    # link_is_pdf or
                    link_is_html
                    or not link_has_extension
                    or link_path == "/"
                    or link_path == ""
                ):
                    # print(f"Skipping discovered link (disallowed extension): {full_url}")
                    continue

                resources_to_crawl.add(full_url)

    print(
        f"Found {len(resources_to_crawl)} valid internal links to consider from {base_url}"
    )

    for resource in resources_to_crawl:
        # The recursive call will handle filtering the resource type at its start
        crawl_webpage(resource, folder, driver)


if __name__ == "__main__":
    list_pages = {
        # "particuliers": "https://www.bgl.lu/fr/particuliers.html",
        "particuliers": "https://bgl.lu/fr/particuliers/contact/localisateur-agences.html",
        # Add more starting points if needed
    }

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")  # Run headless for automation
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--ignore-certificate-errors")
    # Set a realistic user agent
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    )
    # Optional: Disable images/css for faster loading if only HTML is needed
    # chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    # chrome_options.add_experimental_option("prefs", {"profile.managed_default_content_settings.stylesheets": 2})

    # --- IMPORTANT PDF HANDLING NOTE ---
    # Remains the same: Selenium isn't ideal for downloading PDFs.
    # This script identifies them and saves an empty placeholder.
    # Use 'requests' library for actual PDF downloads if needed.
    # ----------------------------------

    driver_service = Service()  # Assumes chromedriver is in PATH or specified
    driver = None
    csvfile_handle = None
    log_filename = "url_log.csv"

    try:
        print("Initializing WebDriver...")
        driver = webdriver.Chrome(service=driver_service, options=chrome_options)
        # Optional: Set a page load timeout for the driver globally
        # This affects how long driver.get() waits before potentially timing out
        driver.set_page_load_timeout(30)  # e.g., 30 seconds

        print("WebDriver initialized.")
        file_exists = os.path.isfile(log_filename)

        with open(log_filename, "a", newline="", encoding="utf-8") as csvfile:
            csvfile_handle = csvfile
            csv_writer = csv.writer(csvfile_handle)

            if not file_exists or os.path.getsize(log_filename) == 0:
                csv_writer.writerow(["Initial URL", "Final URL/Status"])
                csvfile_handle.flush()

            # Store the initial domain for consistent checking
            initial_domain = urlparse(list(list_pages.values())[0]).netloc

            for output_topic, base_url in list_pages.items():
                output_folder = os.path.join("bgl_bnp_resources", output_topic)
                print(f"\n--- Starting crawl for '{output_topic}' ({base_url}) ---")
                crawl_webpage(base_url, output_folder, driver)

    except Exception as e:
        print(f"An critical error occurred during script execution: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
    finally:
        if driver:
            print("Quitting WebDriver...")
            driver.quit()
            print("WebDriver quit.")
        print("Script finished.")
