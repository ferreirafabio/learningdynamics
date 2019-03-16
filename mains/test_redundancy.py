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

    with open('redundancy_log.txt', "a+") as text_file:
        while True:
            try:
                next_element = test_data.get_next_batch()
                features_test = sess.run(next_element)
                while True:
                    try:
                        next_element = train_data.get_next_batch()
                        features_train = sess.run(next_element)
                        print("comparing exp id " + str(features_test["experiment_id"]) + " vs " + str(features_train["experiment_id"]))
                        if np.allclose(features_test['objpos'][0][0], features_train['objpos'][0][0]):
                            print("experiment ids " + str(features_train["experiment_id"]) +"(train) and " +
                                            str(features_test["experiment_id"]) + "(test) have identical objpos\n")

                            text_file.write("experiment ids " + str(features_train["experiment_id"]) +"(train) and " +
                                            str(features_test["experiment_id"]) + "(test) have identical objpos\n")
                    except tf.errors.OutOfRangeError:
                        sess.run(train_data.iterator.initializer)
                        break

            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    main()
