import os
import tensorflow as tf
try:
    [tf.config.experimental.set_memory_growth(
        gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
from mltu.annotations.images import CVImage

from model import train_model
from datetime import datetime

from mltu.configs import BaseModelConfigs

model_name = "model1"


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "models", model_name, datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.height = 50
        self.width = 150
        self.max_text_length = 6
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.train_epochs = 1000
        self.train_workers = 20


dataset, vocab, max_len = [], set(), 0
captcha_path = os.path.join("datasets", model_name)
for file in os.listdir(captcha_path):
    file_path = os.path.join(captcha_path, file)
    label = os.path.splitext(file)[0]
    dataset.append([file_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

configs = ModelConfigs()


configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length,
                     padding_value=len(configs.vocab))
    ],
)

train_data_provider, val_data_provider = data_provider.split(split=0.9)


train_data_provider.augmentors = [
    RandomBrightness(), RandomRotate(), RandomErodeDilate()]


model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
    run_eagerly=False
)
model.summary(line_length=110)
os.makedirs(configs.model_path, exist_ok=True)


earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5",
                             monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")


model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger,
               reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
