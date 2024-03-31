import cv2
import typing
import numpy as np
import os
import time
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder


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


if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load(
        "./model/configs.yaml")

    model = ImageToWordModel(
        model_path=configs.model_path, char_list=configs.vocab)

    success = 0
    real_start = time.time()
    for image in os.listdir('./datasets/captcha1/'):
        img = cv2.imread(f'./datasets/captcha1/{image}')
        now = time.time()
        prediction_text = model.predict(img)
        if prediction_text == image.split(".")[0]:
            success += 1
        print(f"{image} is {prediction_text} predected in {time.time() - now} seconds")
    print(
        f"predicted all {len(os.listdir('./datasets/captcha1/'))} captchas in {time.time() - real_start} seconds with success rate: {(success / len(os.listdir('./datasets/captcha1/')))*100}%")
