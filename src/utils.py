import requests
from PIL import Image


REQUESTS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def download_image_as_pil(url: str) -> Image.Image:
    try:
        response = requests.get(
            url, stream=True, headers=REQUESTS_HEADERS
        )

        if response.status_code == 200:
            return Image.open(response.raw)

    except Exception as e:
        return