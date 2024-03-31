import requests
import base64
import os
from concurrent.futures import ThreadPoolExecutor


def process_image(image):
    with open(f"./scraped/{image}", "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode('utf-8')

    data = {
        "method": "ocr",
        "id": "catch",
        "image": img
    }

    headers = {
        "Content-Type": "application/json",
        "apikey": "fr"
    }

    response = requests.post(
        "https://pro.nocaptchaai.com/solve", json=data, headers=headers)
    response_data = response.json()

    print(response_data)
    try:
        os.rename(
            f"./scraped/{image}", f"./datasets/captcha1/{response_data['solution']}.png")
    except FileExistsError:
        print(f"Error: File '{image}' already exists.")


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        images = os.listdir("./scraped")
        executor.map(process_image, images)
