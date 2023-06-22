import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/03_mc_model", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = "0123456789"
        self.height = 128
        self.width = 128
        self.max_text_length = 5
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.train_epochs = 1000
        self.train_workers = 20