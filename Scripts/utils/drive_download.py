#Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import argparse
import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--file_id', type=str)
    parser.add_argument('-d', '--destination', type=str)

    args = parser.parse_args()
    return args

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    # Command example:
    #   python drive_download.py -i 1A5H4fZ38NxWLyX7wedINghC1iPRb9LzD -d online_retail_II.xlsx
    
    args = parse_args()
    download_file_from_google_drive(args.file_id, args.destination)

