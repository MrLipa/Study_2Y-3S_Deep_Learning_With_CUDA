# src/models/manager.py

import os
import torch
from datetime import datetime
from typing import Optional
from ..utils.singleton import Singleton


class Manager(metaclass=Singleton):
    def __init__(self, model, data_loader, logger, device) -> None:
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.device = device

    def train_model(self, criterion, optimizer: torch.optim.Optimizer, epochs: int) -> None:
        pass

    def save_model(self, models_folder_path: str) -> None:
        os.makedirs(models_folder_path, exist_ok=True)
        current_time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(models_folder_path, f"image_colorizer_{current_time}.pth")
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, model_path: str = "", if_latest: bool = False) -> None:
        path = model_path
        if if_latest:
            path = self._get_latest_model_path(model_path)
        if path:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            self.logger.info(f"Loaded model from {path}")

    def _get_latest_model_path(self, models_folder_path: str) -> Optional[str]:
        model_files = [f for f in os.listdir(models_folder_path) if f.endswith('.pth')]
        if not model_files:
            self.logger.error("No model files found.")
            return None
        latest_model_file = max(model_files, key=lambda x: datetime.strptime(
            x.replace('image_colorizer_', '').replace('.pth', ''), "%Y-%m-%d_%H-%M-%S"))
        return os.path.join(models_folder_path, latest_model_file)

    def predict_model(self):
        pass
