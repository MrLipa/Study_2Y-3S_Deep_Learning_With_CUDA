# src/data/image_net_scraper.py

import os
import requests
import json
from multiprocessing import Pool, cpu_count
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import pandas as pd
from ..utils import Singleton


class ImageNetScraper(metaclass=Singleton):
    def __init__(self, class_list, images_per_class, data_root, multiprocessing_workers, logger):
        self.images_per_class = images_per_class
        self.data_root = data_root
        self.class_list = class_list
        self.multiprocessing_workers = multiprocessing_workers if multiprocessing_workers else cpu_count()
        self.logger = logger

        self.file_path = os.path.dirname(os.path.realpath(__file__))

        self.class_info_dict = self.load_classes_from_json()
        self.class_info_df = self.load_classes_from_csv()

    def load_classes_from_json(self):
        json_filepath = os.path.join(self.file_path, 'imagenet_class_info.json')
        with open(json_filepath) as file:
            return json.load(file)

    def load_classes_from_csv(self):
        csv_filepath = os.path.join(self.file_path, 'classes_in_imagenet.csv')
        return pd.read_csv(csv_filepath)

    def setup_directories(self):
        self.imagenet_images_folder = os.path.join(self.file_path, self.data_root)
        if not os.path.isdir(self.imagenet_images_folder):
            os.mkdir(self.imagenet_images_folder)
            self.logger.info(f"Created directory at {self.imagenet_images_folder}")

    def get_image(self, img_url, class_name, class_images_counter):
        response = requests.get(img_url, timeout=1)
        if 'image' not in response.headers.get('content-type', ''):
            self.logger.error("Not an image")
            raise ValueError("Not an image")
        img_content = response.content
        if len(img_content) < 1000:
            self.logger.error("Image too small")
            raise ValueError("Image too small")
        img_file_path = os.path.join(self.imagenet_images_folder, f'{class_name}_{class_images_counter}.png')
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
        url = f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'
        response = requests.get(url)
        return [url.decode('utf-8') for url in response.content.splitlines()]

    def scrape_class(self, class_wnid):
        class_name = self.class_info_dict[class_wnid]["class_name"]
        img_urls = self.fetch_image_urls(class_wnid)
        self.logger.info(f'Starting download for class "{class_name}" with a limit of {self.images_per_class} images.')
        self.download_images(img_urls, class_name)

    def run(self):
        self.setup_directories()
        with Pool(self.multiprocessing_workers) as pool:
            pool.map(self.scrape_class, self.class_list)
