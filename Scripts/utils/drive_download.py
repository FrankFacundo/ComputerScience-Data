# Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import argparse
import re

import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a file from Google Drive using its shareable link."
        "The file name is automatically extracted from the response headers."
    )
    parser.add_argument(
        "-l", "--link", type=str, required=True, help="Google Drive shareable link"
    )
    return parser.parse_args()


def download_file_from_google_drive_file_id(id):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    destination = get_file_name(response)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def get_file_name(response):
    """
    Extracts file name from the Content-Disposition header.
    Returns a default name if the header is not present.
    """
    cd = response.headers.get("Content-Disposition")
    if cd:
        # Example header: 'attachment; filename="example.pdf"'
        fname = re.findall('filename="(.+)"', cd)
        if fname:
            return fname[0]
    return "downloaded_file"


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(
            response.iter_content(CHUNK_SIZE), desc="Downloading", unit="chunk"
        ):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def extract_file_id(file_input):
    """
    Extracts the file ID from a Google Drive URL. If file_input is not a URL,
    it is assumed to be a file ID and returned as is.

    Example URL: https://drive.google.com/file/d/1IMiuHsiVvna_iN7vdZBMZy8DviFnrvYY/view?usp=sharing
    """
    if file_input.startswith("http"):
        match = re.search(r"/d/([^/]+)", file_input)
        if match:
            return match.group(1)
        else:
            raise ValueError(
                "Invalid Google Drive link format. Could not extract file ID."
            )
    else:
        return file_input


def download_file_from_google_drive(link):
    file_id = extract_file_id(link)
    download_file_from_google_drive_file_id(file_id)


if __name__ == "__main__":
    args = parse_args()
    download_file_from_google_drive(args.link)
