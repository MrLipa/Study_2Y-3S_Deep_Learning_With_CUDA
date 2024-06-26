# src/models/manager.py

import os
import torch
import time
from datetime import datetime
from typing import Optional
from ..utils.singleton import Singleton
import torch.nn.functional as F
from torch.nn.functional import interpolate
import numpy as np
import cv2


class Manager(metaclass=Singleton):
    def __init__(self, model, data_loader, logger, device) -> None:
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.device = device

    def train_model(self, criterion, optimizer: torch.optim.Optimizer, epochs: int) -> None:
        batch_loss = [1000000,1000000,1000000]
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for lab in self.data_loader.train_lab_data_loader:
                L = lab[:, :, :, 0]
                L = L[:, :, :, None]
                L = torch.permute(L, (0, 3, 1, 2))
                input_tensor, ground_truth = L.to(self.device), lab.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_tensor)
                loss = criterion(outputs, ground_truth)
                loss.backward()
                optimizer.step()
                self.logger.info(f"Epoch {epoch}, Batch loss: {loss.item()}")
            epoch_duration = time.time() - epoch_start_time

            val_loss = []
            for lab in self.data_loader.valid_lab_data_loader:
                L = lab[:, :, :, 0]
                L = L[:, :, :, None]
                L = torch.permute(L, (0, 3, 1, 2))
                input_tensor, ground_truth = L.to(self.device), lab.to(self.device)
                outputs = self.model(input_tensor)
                val_loss.append(criterion(outputs, ground_truth).item())
            batch_loss.pop(0)
            batch_loss.append(sum(val_loss) / len(val_loss))
            self.logger.info(f"Epoch {epoch}, Final Loss: {batch_loss[-1]}, Epoch duration: {epoch_duration:.2f} seconds")
            if int(max(batch_loss)) == int(batch_loss[-1]):
                self.logger.info("Early stopping, validation data loss worse than 2 of the epochs before")
                break

    def save_model(self, models_folder_path: str) -> None:
        os.makedirs(models_folder_path, exist_ok=True)
        current_time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(models_folder_path, f"image_colorizer_{current_time}.pth")
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, device, model_path: str = "", if_latest: bool = False) -> None:
        path = model_path
        if if_latest:
            path = self._get_latest_model_path(model_path)
        if path:
            self.model.load_state_dict(torch.load(path, map_location=device))
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

    def _class2ab(self, inClass, scaleFactor):
        aClass = int(inClass / int(256/scaleFactor))
        a = int(aClass * scaleFactor + int(scaleFactor/2))
        bClass = inClass - aClass * int(256/scaleFactor)
        b = int(bClass * scaleFactor + int(scaleFactor/2))
        return a, b

    def predict_model(self, classScaleFactor):
        predictedImages = []
        batchCounter = 0

        for lab in self.data_loader.test_lab_data_loader:
            batchCounter += 1
            L = lab[:, :, :, 0]
            L = L[:, :, :, None]
            L = torch.permute(L, (0, 3, 1, 2))
            input, groudTrouth = L.to(self.device), lab.to(self.device)
            outputs = self.model(input)
            outputs = interpolate(outputs, scale_factor=4, mode='bilinear')
            for image, Lchannel in zip(outputs, L):
                predictedImages.append(np.zeros((256,256,3)))
                for row in range(image.size()[0]):
                    for col in range(image.size()[1]):
                        a, b = self._class2ab(torch.argmax(image[row,col]), classScaleFactor)
                        predictedImages[-1][row,col][0] = Lchannel[0, row, col].item()
                        predictedImages[-1][row,col][1] = a
                        predictedImages[-1][row,col][2] = b
            if batchCounter > 0: #save only batchSize number of images for comparison
                break
        return predictedImages


    def saveImages(self, predictedImages: list):
        for imageIndex in range(len(predictedImages)):
            img = np.float32(predictedImages[imageIndex])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            cv2.imwrite("./data/color_img" + str(imageIndex) + ".jpg", predictedImages[imageIndex])
