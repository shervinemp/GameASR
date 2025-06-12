def download_file(url, destination):
    """
    Download a file from a URL to a specified destination.
    """
    import requests

    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an error for bad responses
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {url} to {destination}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
