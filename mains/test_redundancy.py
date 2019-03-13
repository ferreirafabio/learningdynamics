import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.singulation_trainer import SingulationTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from pydoc import locate
import numpy as np


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

        config.old_tfrecords = args.old_tfrecords
        config.normalize_data = False


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

    while True:
        next_element = test_data.get_next_batch()
        features_test = sess.run(next_element)
        while True:
            next_element = train_data.get_next_batch()
            features_train = sess.run(next_element)
            if np.allclose(features_test['objpos'][0][0], features_train['objpos'][0][0]):
                print("equal")


if __name__ == '__main__':
    main()
