# src/data/image_net_scraper.py

import os
import cv2
import requests
import numpy as np
import json
from multiprocessing import Pool
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import pandas as pd
from ..utils.singleton import Singleton


class ImageNetScraper(metaclass=Singleton):
    def __init__(self, class_list, images_per_class, multiprocessing_workers, data_root, logger):
        self.images_per_class = images_per_class
        self.data_root = data_root
        self.class_list = class_list
        self.multiprocessing_workers = multiprocessing_workers
        self.logger = logger

        self.class_info_dict = self.load_classes_from_json()
        self.class_info_df = self.load_classes_from_csv()

    def load_classes_from_json(self):
        json_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'src', 'resources', 'imagenet_class_info.json')
        with open(json_filepath) as file:
            return json.load(file)

    def load_classes_from_csv(self):
        csv_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'src', 'resources', 'classes_in_imagenet.csv')
        return pd.read_csv(csv_filepath)

    def setup_directories(self):
        if not os.path.isdir(self.data_root):
            os.mkdir(self.data_root)
            self.logger.info(f"Created directory at {self.data_root}")

    def get_image(self, img_url, class_name, class_images_counter):
        if not img_url.startswith("https://"):
            img_url = "https://" + img_url.lstrip("http://")
        response = requests.get(img_url, timeout=0.5)

        if 'image' not in response.headers.get('content-type', ''):
            self.logger.error("Not an image")
            raise ValueError("Not an image")

        img_content = response.content
        if len(img_content) < 1000:
            self.logger.error("Image too small")
            raise ValueError("Image too small")

        nparr = np.frombuffer(img_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            self.logger.error("Failed to decode image or image is empty")
            raise ValueError("Failed to decode image or image is empty")

        img_file_path = os.path.join(self.data_root, f'{class_name}_{class_images_counter}.png')
        with open(img_file_path, 'wb') as img_f:
            img_f.write(img_content)
        self.logger.info(f"Saved image {img_file_path}")

    def download_images(self, urls, class_name):
        class_images_counter = 0

        for url in urls:
            if class_images_counter >= self.images_per_class:
                self.logger.info(f"Reached the limit of {self.images_per_class} images for class {class_name}.")
                break
            try:
                class_images_counter += 1
                self.get_image(url, class_name, class_images_counter)
            except (ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL, ValueError) as e:
                self.logger.error(f"Failed to download image: {e}")
                class_images_counter -= 1

        self.logger.info(f"Downloaded images for class {class_name}.")

    def fetch_image_urls(self, wnid):
        url = f'https://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'
        response = requests.get(url)
        return [url.decode('utf-8') for url in response.content.splitlines()]

    def scrape_class(self, class_wnid):
        class_name = self.class_info_dict[class_wnid]["class_name"]
        img_urls = self.fetch_image_urls(class_wnid)
        self.logger.info(f'Starting download for class "{class_name}" with a limit of {self.images_per_class} images.')
        self.download_images(img_urls, class_name)

    def run(self):
        self.setup_directories()
        if self.multiprocessing_workers:
            with Pool(self.multiprocessing_workers) as pool:
                pool.map(self.scrape_class, self.class_list)
        else:
            for class_ in self.class_list:
                self.scrape_class(class_)
