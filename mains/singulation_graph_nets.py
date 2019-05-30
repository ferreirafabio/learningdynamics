import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.singulation_trainer_predictor import SingulationTrainerPredictor
from trainers.singulation_trainer import SingulationTrainer
from trainers.singulation_trainer_predictor_extended import SingulationTrainerPredictorExtended
from trainers.singulation_trainer_predictor_extended_objpos import SingulationTrainerPredictorExtendedObjPosition
from trainers.singulation_trainer_auto_encoding import SingulationTrainerAutoEncoder
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from pydoc import locate

import matplotlib
matplotlib.use('Agg')


def main():
    tf.set_random_seed(12345)
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

        if not hasattr(config, 'do_multi_step_prediction'):
            config.do_multi_step_prediction = False

        if not hasattr(config, 'remove_edges'):
            config.remove_edges = False

        if not hasattr(config, 'batch_processing'):
            config.batch_processing = False

        if not hasattr(config, 'conv_layer_instance_norm'):
            config.conv_layer_instance_norm = True
            print("convolution layers are normalized with instance or batch norm")

        if not hasattr(config, 'use_baseline_auto_predictor'):
            config.use_baseline_auto_predictor = False

        if not hasattr(config, 'nodes_get_full_rgb_depth'):
            config.nodes_get_full_rgb_depth = False

        if not hasattr(config, 'n_predictions'):
            config.n_predictions = 1

        if not hasattr(config, 'remove_pos_vel'):
            config.remove_pos_vel = False

        if not hasattr(config, 'edges_carry_segmentation_data'):
            config.edges_carry_segmentation_data = False

        if not hasattr(config, 'use_f_interact'):
            config.use_f_interact = False

        if config.normalize_data:
            print("-- using normalized data as input --")
        else:
            print("-- using unnormalized data as input --")

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
        only_test = True

    # create trainer and pass all the previous components to it
    if "baseline_auto_predictor_extended_multistep_position" in config.model_zoo_file:
        trainer = SingulationTrainerPredictorExtendedObjPosition(sess, model, train_data, test_data, config, logger, only_test=False)
    elif "predictor_extended" in config.model_zoo_file:
        trainer = SingulationTrainerPredictorExtended(sess, model, train_data, test_data, config, logger, only_test=False)
    elif "predictor_" in config.model_zoo_file:
        trainer = SingulationTrainerPredictor(sess, model, train_data, test_data, config, logger, only_test=False)
    elif "auto_encoder" in config.model_zoo_file:
        trainer = SingulationTrainerAutoEncoder(sess, model, train_data, test_data, config, logger, only_test=False)
    else:
        trainer = SingulationTrainer(sess, model, train_data, test_data, config, logger, only_test=only_test)
        print(" --- only initializing test graph --- ")

    # load model if exists
    model.load(trainer.sess)

    if config.mode == "train_test":
        print("--- Running TRAIN/TEST MODE ---")
        trainer.train()
    elif config.mode == "test":
        print("--- Running TEST MODE ---")
        trainer.test()
    elif config.mode == "compute_metrics_over_test_set":
        print("--- Running METRIC COMPUTATION OVER TEST SET ---")
        trainer.compute_metrics_over_test_set()
    elif config.mode == "compute_metrics_over_test_set_multistep":
        print("--- Running METRIC COMPUTATION OVER TEST SET (MULTISTEP) ---")
        trainer.compute_metrics_over_test_set_multistep()
    elif config.mode == "test_specific_exp_ids":
        print("--- Running SPECIFIC EXP ID'S TEST MODE ---")
        trainer.test_specific_exp_ids()
    elif config.mode == "store_latent_vectors":
        print("--- Running store latent vectors mode ---")
        trainer.store_latent_vectors()
    elif config.mode == "save_encoder_vectors":
        print("--- Running SAVE ENCODER VECTORS ---")
        trainer.save_encoder_vectors(train=False)


if __name__ == '__main__':
    main()
