# src/utils/logger.py

import logging
import os
from .singleton import Singleton


class Logger(metaclass=Singleton):
    def __init__(self, name, level, log_directory, log_file):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        fh = logging.FileHandler(os.path.join(log_directory, log_file))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
