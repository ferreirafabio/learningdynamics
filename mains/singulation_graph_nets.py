import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.singulation_graph import create_singulation_graph_nx, create_graph_and_get_graph_ph, create_n_singulation_graphs
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


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
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)

    next_element = data.iterator.get_next()

    for _ in range(100):
        sess.run(data.iterator.initializer)
        while True:
            try:
                #img, seg, gripperpos, objpos, objvel, obj_segs, experiment_length, experiment_id, n_total_objects, n_manipulable_objects
                #  = \
                #    sess.run(next_element)

                obj_segs = sess.run(next_element)
                #print("experiment_length", experiment_length)
                #print("objpos", objpos)
            except tf.errors.OutOfRangeError:
                break

    # todo: get these attributes from tfrecords
    n_manipulable_objects = 3
    n_total_objects = 5
    experiment_length = 20

    # create an instance of the model you want
    base_graphs = create_n_singulation_graphs(experiment_length, config, n_total_objects, n_manipulable_objects)


    graph_ph = create_graph_and_get_graph_ph(config, n_total_objects, n_manipulable_objects)


    print(graph_ph)

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