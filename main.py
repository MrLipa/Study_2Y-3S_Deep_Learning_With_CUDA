import os
import sys
import logging
import torch
from time import time
from src import utils, data, models

project_dir = os.path.abspath('.')
sys.path.append(project_dir)


def main():
    start_time = time()

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    logger = utils.Logger(name='image_colorizer', level=logging.INFO, log_directory='./logs',
                          log_file='log_file.log').get_logger()
    logger.info(f"Session start on {device}")

    input_filepath = './data'
    split_proportions = (0.7, 0.2, 0.1)
    image_size = (256, 256)
    logger = logger
    # class_list = ['n00006484', 'n00007846', 'n00440382', 'n00445055', 'n00447540']
    class_list = ['n00006484']
    images_per_class = 70
    multiprocessing_workers = 5
    batch_size = 10
    epochs = 30
    numberOfBins = 16
    loss_function = utils.LossFunction(numberOfBins)

    loader = data.Loader(input_filepath=input_filepath, split_proportions=split_proportions, image_size=image_size,
                         logger=logger, class_list=class_list, images_per_class=images_per_class,
                         multiprocessing_workers=multiprocessing_workers, batch_size=batch_size, loss_function=loss_function)

    # loader.clear_directories()
    loader.setup_paths()
    # loader.load_and_split_data()
    loader.setup_data_loaders()

    logger.info(f"Load data completed in {time() - start_time:.2f} seconds")

    model = models.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = models.Manager(model, loader, logger, device)
    manager.train_model(loss_function, optimizer, epochs)
    manager.save_model("./models/")

    predicted = manager.predict_model(int(image_size[0]/numberOfBins))
    manager.saveImages(predicted)

    logger.info(f"Session completed in {time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
