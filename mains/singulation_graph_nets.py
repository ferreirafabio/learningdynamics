import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.singulation_trainer import SingulationTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from pydoc import locate

import matplotlib
matplotlib.use('Agg')


def main():
    tf.set_random_seed(1234)
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

        config.old_tfrecords = args.old_tfrecords
        config.normalize_data = args.normalize_data

        if args.n_epochs:
            print("overwriting n_epochs in config file")
            config.n_epochs = args.n_epochs

        if args.mode is not None:
            print("overwriting mode in config file")
            config.mode = args.mode

        if args.tfrecords_dir is not None:
            print("overwriting tfrecord dir in config file")
            config.mode = args.mode

        if not hasattr(config, 'latent_state_noise'):
            config.latent_state_noise = False

        if config.normalize_data:
            print("-- using normalized data as input --")
        else:
            print("-- using unnormalized data as input --")


        # model = import_class_by_string("models.model_zoo." + config.model_zoo_file)
        model_class = locate("models.model_zoo." + config.model_zoo_file + "." + config.model_zoo_file)

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

    model = model_class(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    if config.mode.startswith("train"):
        only_test = False
    else:
        print(" --- only initializing test graph --- ")
        only_test = True

    # create trainer and pass all the previous components to it
    trainer = SingulationTrainer(sess, model, train_data, test_data, config, logger, only_test=only_test)

    # load model if exists
    model.load(sess)

    if config.mode == "train_test":
        print("--- Running TRAIN/TEST MODE ---")
        trainer.train()
    elif config.mode == "test":
        print("--- Running TEST MODE ---")
        trainer.test_rollouts()
    elif config.mode == "test_specific_exp_ids":
        print("--- Running SPECIFIC EXP ID'S TEST MODE ---")
        trainer.test_specific_exp_ids()
    elif config.mode == "test_5_objects_one_step":
        print("--- Running one-step 5 object test mode ---")
        trainer.test_5_objects_one_step()
    elif config.mode == "test_5_objects_multi_step":
        print("--- Running multi-step 5 object test mode ---")
        trainer.test_5_objects_multi_step()
    elif config.mode == "test_statistics":
        print("--- Running STATISTICAL TEST MODE ---")
        trainer.test_statistics(prefix=config.exp_name, initial_pos_vel_known=config.initial_pos_vel_known,
                                export_latent_data=False, sub_dir_name="test_stats")

if __name__ == '__main__':
    main()