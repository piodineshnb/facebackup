import requests
import time

def download_file_from_url(url, path, retries=5):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP request errors
            with open(path, 'wb') as f:
                f.write(response.content)
            return url.split('.')[-1]  # Return the extension of the file
        except requests.exceptions.RequestException as e:
            if attempt == retries - 2:  # Check if it's the second last attempt
                time.sleep(1)  # Delay for 1 second
            if attempt == retries - 1:
                raise Exception(f"Failed to download file after {retries} attempts: {e} for url: {url}")
