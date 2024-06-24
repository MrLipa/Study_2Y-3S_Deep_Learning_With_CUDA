# src/data/loader.py

import torch
import os
import shutil
import cv2
from torchvision import transforms
from .image_net_scraper import ImageNetScraper
from ..utils.singleton import Singleton
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, directory, transform, image_type):
        self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if image_type == 'original':
            self.code = cv2.COLOR_BGR2RGB
        elif image_type == 'lab':
            self.code = cv2.COLOR_BGR2LAB

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, self.code)
        if self.transform:
            image = self.transform(image)
        return image


class Loader(metaclass=Singleton):
    def __init__(self, input_filepath, split_proportions, image_size, batch_size, logger, class_list, images_per_class, multiprocessing_workers, loss_function):
        self.input_filepath = input_filepath
        self.split_proportions = split_proportions
        self.image_size = image_size
        self.logger = logger
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.image_net_scraper = ImageNetScraper(class_list=class_list, images_per_class=images_per_class, data_root=input_filepath, multiprocessing_workers=multiprocessing_workers, logger=logger)

        os.makedirs(self.input_filepath, exist_ok=True)

    def setup_paths(self):

        types = ['original', 'lab']
        stages = ['train', 'valid', 'test']
        self.paths = {}
        for t in types:
            self.paths[t] = {}
            for stage in stages:
                dir_path = os.path.join(self.input_filepath, f"{t}/{stage}")
                self.paths[t][stage] = dir_path
                os.makedirs(dir_path, exist_ok=True)
        self.logger.info("Directories set up for original, lab, and gray images.")

    def load_and_split_data(self):
        self.logger.info("Downloading images.")
        self.image_net_scraper.run()
        images = [os.path.join(self.input_filepath, f) for f in os.listdir(self.input_filepath) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.distribute_images(images)

    def distribute_images(self, images):
        total_images = len(images)
        indices = torch.randperm(total_images)
        train_end = int(total_images * self.split_proportions[0])
        valid_end = train_end + int(total_images * self.split_proportions[1])

        self.process_images(images, indices[:train_end], 'train')
        self.process_images(images, indices[train_end:valid_end], 'valid')
        self.process_images(images, indices[valid_end:], 'test')

    def process_images(self, images, indices, category):
        for idx in indices:
            image_path = images[idx]
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Failed to load image: {image_path}")
                continue

            try:
                lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                self.save_original_image(image_path, category)
                self.save_lab_image(lab_image, image_path, category)
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")

    def save_original_image(self, img_path, category):
        destination = self.paths['original'][category]
        shutil.move(img_path, os.path.join(destination, os.path.basename(img_path)))

    def save_lab_image(self, lab_image, img_path, category):
        cv2.imwrite(os.path.join(self.paths['lab'][category], os.path.basename(img_path)), lab_image)

    def setup_data_loaders(self):
        transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.resize(x, (256, 256)))
        ])

        for t in ['original', 'lab']:
            for stage in ['train', 'valid', 'test']:
                dataset = CustomDataset(self.paths[t][stage], transform, image_type=t)
                setattr(self, f"{stage}_{t}_data_loader", DataLoader(dataset, batch_size=self.batch_size, shuffle=True))

        self.logger.info("Data loaders set up for all categories and stages.")

    def clear_directories(self):
        shutil.rmtree(self.input_filepath)
        os.makedirs(self.input_filepath, exist_ok=True)
        self.logger.info("Cleared and reset directory")

    @staticmethod
    def display_first_image(data_loader):
        data_iter = iter(data_loader)
        image = next(data_iter)

        image = image[0]
        image = image.numpy()

        plt.imshow(image)
        plt.axis('off')
        plt.show()
