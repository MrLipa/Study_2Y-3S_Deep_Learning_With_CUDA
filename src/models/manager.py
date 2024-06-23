# src/models/manager.py

import os
import torch
from datetime import datetime
from typing import Optional
from ..utils.singleton import Singleton
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
        for epoch in range(epochs):
            for lab in self.data_loader.train_lab_data_loader:
                L = lab[:,:,:,0]                    # get L channel to for input
                L = L[:, :, :, None]                # add 1 as channel dimention
                L = torch.permute(L, (0,3,1,2))     # permute batch to shape (batch_szie, channels, H, W)
                input, groudTrouth = L.to(self.device), lab.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input)
                loss = criterion(outputs, groudTrouth)
                print(f"loss: {loss}")
                loss.backward()
                print("backward")
                optimizer.step()
            self.logger.info(f"Epoch {epoch}, Loss: {loss}")

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

    def _class2ab(self, inClass, scaleFactor):
        aClass = int(inClass / int(256/scaleFactor))
        a = int(aClass * scaleFactor + int(scaleFactor/2))
        bClass = inClass - aClass * int(256/scaleFactor)
        b = int(bClass * scaleFactor + int(scaleFactor/2))
        return a, b

    def predict_model(self, scaleFactor):
        counter = 0
        predictedImages = []
        for lab in self.data_loader.train_lab_data_loader:
            L = lab[:,:,:,0]                    # get L channel to for input
            L = L[:, :, :, None]                # add 1 as channel dimention
            L = torch.permute(L, (0,3,1,2))     # permute batch to shape (batch_szie, channels, H, W)
            input, groudTrouth = L.to(self.device), lab.to(self.device)
            outputs = self.model(input)
            outputs = interpolate(outputs, scale_factor=4, mode='bilinear')
            for image, Lchannel in zip(outputs, L):
                predictedImages.append(np.zeros((256,256,3)))
                for row in range(image.size()[0]):
                    for col in range(image.size()[1]):
                        a, b = self._class2ab(torch.argmax(image[row,col]), scaleFactor)
                        predictedImages[-1][row,col][0] = Lchannel[0, row, col].item()
                        predictedImages[-1][row,col][1] = a
                        predictedImages[-1][row,col][2] = b
                if counter > 10: #save only 10 images for comparison
                    break
                counter += 1
        return predictedImages

    def saveImages(self, predictedImages: list):
        for imageIndex in range(len(predictedImages)):
            img = np.float32(predictedImages[imageIndex])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            cv2.imwrite("./data/color_img" + str(imageIndex) + ".jpg", predictedImages[imageIndex])



