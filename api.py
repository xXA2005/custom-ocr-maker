from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from mltu.configs import BaseModelConfigs
import typing
import time
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

app = Flask(__name__)


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


model = None


def load_model():
    global model
    configs = BaseModelConfigs.load("./model/configs.yaml")
    model = ImageToWordModel(
        model_path=configs.model_path, char_list=configs.vocab)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            load_model()

        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'no image'}), 400

        base64_image = data['image']
        image_bytes = base64.b64decode(base64_image)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        now = time.time()
        prediction_text = model.predict(img)
        duration = time.time() - now

        return jsonify({'prediction': prediction_text, 'duration': f'{duration* 1000}ms'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5555, host="0.0.0.0")
