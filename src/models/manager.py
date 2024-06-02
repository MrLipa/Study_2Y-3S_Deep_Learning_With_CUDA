# src/models/manager.py

import mlflow
import os
import torch
import torch.nn as nn
from ..data import Loader
from ..utils.logger import Logger


class Manager:
    def __init__(self, model: nn.Module, data_loader: Loader, logger: Logger = None,
                 mlflow_enabled: bool = False, mlflow_path: str = './../mlruns',
                 experiment_name: str = 'Image Colorizator') -> None:
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.mlflow_enabled = mlflow_enabled
        self.mlflow_path = mlflow_path
        self.experiment_name = experiment_name

        if self.mlflow_enabled:
            self.init_mlflow()

    def init_mlflow(self):
        if not os.path.exists(self.mlflow_path):
            os.makedirs(self.mlflow_path, exist_ok=True)

        mlflow.set_tracking_uri(self.mlflow_path)
        mlflow.set_experiment(self.experiment_name)

        self.logger.info(f"MLflow configured with tracking URI: {self.mlflow_path} and experiment name: {self.experiment_name}")

    def train_model(self, criterion, optimizer, epochs: int):
        pass

    def _train_model(self, criterion, optimizer, epochs: int):
        pass

    def save_model(self, models_folder_path: str):
        pass

    def load_model(self, model_path: str = "", if_latest: bool = False):
        pass

    def delete_model_folder(self, models_folder_path: str):
        pass

    def predict_model(self, images_tensor: torch.Tensor):
        pass
