import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.singulation_graph import create_placeholders, create_feed_dict
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args, convert_dict_to_list_subdicts
from models.loss_functions import create_loss_ops
from models.singulation_models import EncodeProcessDecode


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
    n_epochs = config.n_epochs
    train_batch_size = config.train_batch_size
    global_output_size = config.global_output_size
    edge_output_size = config.edge_output_size
    node_output_size = config.node_output_size
    message_passing_steps = config.message_passing_steps
    learning_rate = config.learning_rate

    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    train_data = DataGenerator(config, train=True)
    next_element = train_data.iterator.get_next()

    optimizer = tf.train.AdamOptimizer(learning_rate)

    # create an instance of the model you want
    model = EncodeProcessDecode(node_output_size=node_output_size, edge_output_size=edge_output_size,
                                       global_output_size=global_output_size)

    # create tensorboard logger
    logger = Logger(sess, config)

    for _ in range(n_epochs):
        sess.run(train_data.iterator.initializer)
        for i in range(train_data.iterations_per_epoch):
            features_dict = sess.run(next_element)
            batch_list = convert_dict_to_list_subdicts(features_dict, train_batch_size)
            input_phs, target_phs, input_graphs, target_graphs = create_placeholders(config=config, batch_data=batch_list, batch_size=train_batch_size)


            for j in range(train_batch_size):
                input_ph = input_phs[j]
                target_ph = target_phs[j]
                input_graph = input_graphs[j]
                target_graph = target_graphs[j]

                exp_length = batch_list[j]['experiment_length']
                output_ops_train = model(input_ph, exp_length)
                loss_ops_tr = create_loss_ops(target_ph, output_ops_train)
                loss_op_tr = sum(loss_ops_tr) / exp_length
                step_op = optimizer.minimize(loss_op_tr)

                # try:
                #     sess.close()
                # except NameError:
                #     pass
                #
                # sess = tf.Session()

                sess.run(tf.global_variables_initializer())
                feed_dict = {input_ph: input_graph, target_ph: target_graph}
                train_values = sess.run({"step": step_op, "target": target_ph, "loss": loss_op_tr}, feed_dict=feed_dict)
                print(train_values['loss'])

            # todo: train/test cycles





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