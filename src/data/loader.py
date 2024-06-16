# src/data/loader.py

import torch
import os
import shutil
import cv2
from torchvision import transforms
from .image_net_scraper import ImageNetScraper
from ..utils.singleton import Singleton
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image


class Loader(metaclass=Singleton):
    def __init__(self, input_filepath, split_proportions, image_size, logger, class_list, images_per_class, multiprocessing_workers, batch_size):
        self.input_filepath = input_filepath
        self.split_proportions = split_proportions
        self.image_size = image_size
        self.logger = logger
        self.batch_size = batch_size

        self.image_net_scraper = ImageNetScraper(class_list=class_list, images_per_class=images_per_class, data_root=input_filepath, multiprocessing_workers=multiprocessing_workers, logger=logger)

        self.train_path = os.path.join(self.input_filepath, 'train')
        self.valid_path = os.path.join(self.input_filepath, 'valid')
        self.test_path = os.path.join(self.input_filepath, 'test')

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def setup_directories(self):
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        self.logger.info("Created directories")

    def load_and_split_data(self):
        self.logger.info("Starting to run scraper to download images.")
        self.image_net_scraper.run()

        images = [os.path.join(self.input_filepath, file) for file in os.listdir(self.input_filepath) if file.endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)
        self.logger.info(f"Found {total_images} images for processing.")

        train_size = int(total_images * self.split_proportions[0])
        valid_size = int(total_images * self.split_proportions[1])

        torch.manual_seed(0)
        shuffled_indices = torch.randperm(total_images)

        for idx in shuffled_indices[:train_size]:
            shutil.move(images[idx], os.path.join(self.train_path, os.path.basename(images[idx])))
            self.logger.info(f"Moved {images[idx]} to {self.train_path}")

        for idx in shuffled_indices[train_size:train_size + valid_size]:
            shutil.move(images[idx], os.path.join(self.valid_path, os.path.basename(images[idx])))
            self.logger.info(f"Moved {images[idx]} to {self.valid_path}")

        for idx in shuffled_indices[train_size + valid_size:]:
            shutil.move(images[idx], os.path.join(self.test_path, os.path.basename(images[idx])))
            self.logger.info(f"Moved {images[idx]} to {self.test_path}")

    def setup_data_loaders(self):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.resize(x, (256, 256)))
        ])

        self.train_data = DataLoader(CustomDataset(self.train_path, transform=transform), batch_size=self.batch_size, shuffle=True)
        self.valid_data = DataLoader(CustomDataset(self.valid_path, transform=transform), batch_size=self.batch_size, shuffle=True)
        self.test_data = DataLoader(CustomDataset(self.test_path, transform=transform), batch_size=self.batch_size, shuffle=True)

        self.logger.info("DataLoaders for train, valid, and test are now set up.")

    def clear_directories(self):
        directory = self.input_filepath
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Cleared and reset directory {directory}")
