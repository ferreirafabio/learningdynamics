import tensorflow as tf


from data_loader.data_generator import DataGenerator
from trainers.singulation_mpc import MPCTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from models.model_zoo.old.EncodeProcessDecode_v2 import EncodeProcessDecode_v2


def main():
  # capture the config path from the run arguments
  # then process the json configuration file
  try:
    args = get_args()
    config = process_config(args.config)

  except Exception as e:
    print("An error occurred during processing the configuration file")
    print(e)
    exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.config_file_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    train_data = DataGenerator(config, sess, old_tfrecords=config.old_tfrecords, train=True)
    test_data = DataGenerator(config, sess, old_tfrecords=config.old_tfrecords, train=False)

    model = EncodeProcessDecode_v2(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = MPCTrainer(sess, model, train_data, test_data, config, logger, N=10)

    # load model if exists
    model.load(sess)

    results = trainer.mpc()
