# src/models/manager.py

import mlflow
import os
import torch
import torchvision

from . import Model
from ..utils import Singleton


class Manager(metaclass=Singleton):
    def __init__(self, model, data_loader, logger, mlflow_enabled, mlflow_path, experiment_name) -> None:
        self.project_path = os.environ.get('PROJECT_PATH')
        print("aaa")
        self.model = model
        self.data_loader = data_loader
        self.logger = logger

        self.mlflow_enabled = mlflow_enabled
        self.mlflow_path = os.path.join(self.project_path, mlflow_path)
        self.experiment_name = experiment_name

        self.train_images = []
        self.test_images = []
        self.validation_images = []

        for file in os.listdir(self.data_loader.processed_train_path):
            print(file)
            path = os.path.join(self.data_loader.processed_train_path, file)
            print(path)
            img = torchvision.io.read_image(path)
            self.train_images.append(img)

        for file in os.listdir(self.data_loader.processed_test_path):
            print(file)
            path = os.path.join(self.data_loader.processed_test_path, file)
            print(path)
            img = torchvision.io.read_image(path)
            self.test_images.append(img)

        for file in os.listdir(self.data_loader.processed_valid_path):
            print(file)
            path = os.path.join(self.data_loader.processed_valid_path, file)
            print(path)
            img = torchvision.io.read_image(path)
            self.validation_images.append(img)

        if self.mlflow_enabled:
            self.init_mlflow()

    def init_mlflow(self):
        if not os.path.exists(self.mlflow_path):
            os.makedirs(self.mlflow_path, exist_ok=True)

        mlflow.set_tracking_uri(self.mlflow_path)
        mlflow.set_experiment(self.experiment_name)

        self.logger.info(f"MLflow configured with tracking URI: {self.mlflow_path} and experiment name: {self.experiment_name}")

    def train_model(self):
        opt = torch.optim.Adam(self.model.parameters())

        self.data_loader.processed_train_path = os.path.join(self.data_loader.output_filepath, 'train')
        self.data_loader.processed_valid_path = os.path.join(self.data_loader.output_filepath, 'valid')
        self.data_loader.processed_test_path = os.path.join(self.data_loader.output_filepath, 'test')

        opt.zero_grad(True)
        pred = self.model(self.train_images[0])
        loss = torch.nn.CrossEntropyLoss()(pred, self.train_images[0])
        loss.backward()
        opt.step()

    def save_model(self, model_path):
        # torch.save(self.model.state_dict(), model_path)

        torch.save(self.model, model_path)

    def load_model(self, model_path):
        # self.model = Model(*args, **kwargs)
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        self.model = torch.load(model_path)

    def delete_model_folder(self):
        pass

    def predict_model(self):
        pass
