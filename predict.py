import os
import sys
import logging
import torch
from time import time

project_dir = os.path.abspath('.')
sys.path.append(project_dir)

from src import utils, data, models


def main():
    start_time = time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = utils.Logger(name='image_colorizer', level=logging.INFO, log_directory=os.path.join(project_dir, 'logs'),
                          log_file='log_file.log').get_logger()
    logger.info(f"Session start on {device}")

    input_filepath = os.path.join(project_dir, 'data')
    split_proportions = (0.7, 0.2, 0.1)
    image_size = (256, 256)
    logger = logger
    class_list = ['n00448232']
    images_per_class = 1000
    multiprocessing_workers = 5
    batch_size = 32
    numberOfBins = 16
    loss_function = utils.LossFunction(numberOfBins)

    loader = data.Loader(input_filepath=input_filepath, split_proportions=split_proportions, image_size=image_size,
                         logger=logger, class_list=class_list, images_per_class=images_per_class,
                         multiprocessing_workers=multiprocessing_workers, batch_size=batch_size, loss_function=loss_function)

    loader.setup_paths()
    loader.setup_data_loaders()

    logger.info(f"Load data completed in {time() - start_time:.2f} seconds")

    model = models.Model().to(device)
    manager = models.Manager(model, loader, logger, device)
    manager.load_model(device, model_path = "./src/models/", if_latest = True)
    predictions = manager.predict_model(int(image_size[0]/numberOfBins))
    manager.saveImages(predictions)
    logger.info(f"Session completed in {time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
