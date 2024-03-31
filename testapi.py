import base64
import requests
import os


def test_api(image_path):
    with open(image_path, 'rb') as f:
        img = f.read()

    img_base64 = base64.b64encode(img).decode()

    payload = {'image': img_base64}

    api_url = 'http://localhost:5555/predict'
    response = requests.post(api_url, json=payload, headers={
                             'Content-Type': 'application/json'})

    return response.json()


if __name__ == '__main__':
    for img in os.listdir('./testing/'):
        r = test_api(f'./testing/{img}')
        print(r)
