import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.singulation_graph import generate_singulation_graph, create_graph_and_get_graph_ph, create_placeholders, get_graph_tuple
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args, convert_dict_to_list_subdicts
from graph_nets.demos import models
from graph_nets import utils_tf, utils_np


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    n_epochs = config["n_epochs"]
    train_batch_size = config["train_batch_size"]

    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    next_element = data.iterator.get_next()

    for _ in range(n_epochs):
        sess.run(data.iterator.initializer)
        while True:
            try:
                features_dict = sess.run(next_element)
                batch_list = convert_dict_to_list_subdicts(features_dict, train_batch_size)
                input_graphs, target_graphs = create_placeholders(config=config, batch_data=batch_list, train_batch_size=train_batch_size)

                # instantiate model
                model = models.EncodeProcessDecode()
                # todo: train/test cycles


                #ase_graphs = create_placeholders(int(experiment_length), config, int(n_total_objects), int(n_manipulable_objects))

                #graph_ph = create_graph_and_get_graph_ph(config, int(n_total_objects), int(n_manipulable_objects))
                #graph_nx = generate_singulation_graph(config, int(n_total_objects), int(n_manipulable_objects))

                #graph_tuple = get_graph_tuple(graph_nx)
                #feed_dict = utils_tf.get_feed_dict(graph_ph, graph_tuple)


            except tf.errors.OutOfRangeError:
                break




    # create tensorboard logger
    #logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    #trainer = ExampleTrainer(sess, graph_ph, data, config, logger)
    # load model if exists
    #model.load(sess)
    # here you train your model
    #trainer.train()


if __name__ == '__main__':
    main()