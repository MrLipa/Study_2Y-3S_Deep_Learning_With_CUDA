# src/models/manager.py

import mlflow
import os
import torch

from . import Model
from ..utils import Singleton


class Manager(metaclass=Singleton):
    def __init__(self, model, data_loader, logger, mlflow_enabled, mlflow_path, experiment_name) -> None:
        self.project_path = os.environ.get('PROJECT_PATH')

        self.model = model
        self.data_loader = data_loader
        self.logger = logger

        self.mlflow_enabled = mlflow_enabled
        self.mlflow_path = os.path.join(self.project_path, mlflow_path)
        self.experiment_name = experiment_name

        if self.mlflow_enabled:
            self.init_mlflow()

    def init_mlflow(self):
        if not os.path.exists(self.mlflow_path):
            os.makedirs(self.mlflow_path, exist_ok=True)

        mlflow.set_tracking_uri(self.mlflow_path)
        mlflow.set_experiment(self.experiment_name)

        self.logger.info(f"MLflow configured with tracking URI: {self.mlflow_path} and experiment name: {self.experiment_name}")

    def train_model(self):
        pass

    def _train_model(self):
        pass

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
