# src/data/loader.py

import torch
import os
from torchvision import transforms
import shutil
import cv2
from .image_net_scraper import ImageNetScraper
from ..utils import Singleton


class Loader(metaclass=Singleton):
    def __init__(self, input_filepath, output_filepath, split_proportions, image_size):
        self.file_path = os.path.dirname(os.path.realpath(__file__))

        self.input_filepath = os.path.join(self.file_path, input_filepath)
        self.output_filepath = os.path.join(self.file_path, output_filepath)
        self.split_proportions = split_proportions
        self.image_size = image_size

        self.image_net_scraper = ImageNetScraper(['n00006484', 'n00007846', 'n00440382', 'n00445055', 'n00447540'], images_per_class=2, data_root=input_filepath, multiprocessing_workers=5)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def setup_directories(self):
        self.train_path = os.path.join(self.input_filepath, 'train')
        self.valid_path = os.path.join(self.input_filepath, 'valid')
        self.test_path = os.path.join(self.input_filepath, 'test')

        self.processed_train_path = os.path.join(self.output_filepath, 'train')
        self.processed_valid_path = os.path.join(self.output_filepath, 'valid')
        self.processed_test_path = os.path.join(self.output_filepath, 'test')

        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        os.makedirs(self.processed_train_path, exist_ok=True)
        os.makedirs(self.processed_valid_path, exist_ok=True)
        os.makedirs(self.processed_test_path, exist_ok=True)

    def load_and_split_data(self):
        self.image_net_scraper.run()

        images = [os.path.join(self.input_filepath, file) for file in os.listdir(self.input_filepath) if file.endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)

        train_size = int(total_images * self.split_proportions[0])
        valid_size = int(total_images * self.split_proportions[1])

        torch.manual_seed(0)
        shuffled_indices = torch.randperm(total_images)

        for idx in shuffled_indices[:train_size]:
            shutil.move(images[idx], os.path.join(self.train_path, os.path.basename(images[idx])))

        for idx in shuffled_indices[train_size:train_size + valid_size]:
            shutil.move(images[idx], os.path.join(self.valid_path, os.path.basename(images[idx])))

        for idx in shuffled_indices[train_size + valid_size:]:
            shutil.move(images[idx], os.path.join(self.test_path, os.path.basename(images[idx])))

    def process_data(self):
        for data_type in ['train', 'valid', 'test']:
            input_dir = os.path.join(self.input_filepath, data_type)
            output_dir = os.path.join(self.output_filepath, data_type)
            files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for file in files:
                file_path = os.path.join(input_dir, file)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                image = cv2.resize(image, self.image_size)

                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, image)

    def clear_directories(self):
        for directory in [self.input_filepath, self.output_filepath]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
        print("Directories have been cleared and reset.")
