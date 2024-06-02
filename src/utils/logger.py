import logging
import os


class Logger:
    def __init__(self, name, level=logging.INFO, log_directory='logs'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        fh = logging.FileHandler(os.path.join(log_directory, 'log_file.log'))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
