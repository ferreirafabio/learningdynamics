import tensorflow as tf
import argparse

from data_loader.data_generator import DataGenerator
from trainers.singulation_trainer import SingulationTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from models.singulation_models import EncodeProcessDecode


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

        if args.n_epochs:
            print("overwriting n_epochs in config file")
            config.n_epochs = args.n_epochs

        if args.mode:
            print("overwriting mode in config file")
            config.mode = args.mode

    except Exception as e:
        print("An error occurred during processing the configuration file")
        print(e)
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.config_file_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    train_data = DataGenerator(config, sess, train=True)
    test_data = DataGenerator(config, sess, train=False)

    model = EncodeProcessDecode(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = SingulationTrainer(sess, model, train_data, test_data, config, logger)

    # load model if exists
    model.load(sess)

    if config.mode == "train_test":
        print("--- Running TRAIN/TEST MODE ---")
        trainer.train()
    elif config.mode == "test":
        print("--- Running TEST MODE ---")
        trainer.test_rollouts()

if __name__ == '__main__':
    main()