import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.constants as constants
from itertools import product
from graph_nets import utils_tf, utils_np


def generate_singulation_graph(config, n_total_objects, n_manipulable_objects):
    graph_nx = nx.MultiDiGraph()

    """ adding features to nodes """
    # 120*160*4(seg+rgb)
    graph_nx.add_node(0, type_name='container')
    graph_nx.add_node(0, features=np.zeros(shape=(134406,), dtype=np.float32))
    #graph_nx.add_node(0, object_segments=np.ndarray(shape=(134400,), dtype=np.int16))

    # 120*160*4(seg+rgb)+3(pos)
    graph_nx.add_node(1, type_name='gripper')
    graph_nx.add_node(1, features=np.zeros(shape=(134406,), dtype=np.float32))
    #graph_nx.add_node(1, object_segments=np.ndarray(shape=(134400,), dtype=np.int16))
    #graph_nx.add_node(1, gripperpos=np.ndarray(shape=(3,), dtype=np.float32))


    for i in range(n_manipulable_objects):
        ''' no multiple features -> flatten and concatenate everything '''
        # 120*160*4(seg+rgb)+3(pos)+3(vel)
        object_str = str(i)
        i = i + 2 # number of non-manipulable objects: 2
        graph_nx.add_node(i, type_name='manipulable_object_' + object_str)
        graph_nx.add_node(i, features=np.zeros(shape=(134406,), dtype=np.float32))
        #graph_nx.add_node(i, object_segments=np.ndarray(shape=(134400,), dtype=np.int16))
        #graph_nx.add_node(i, objpos=np.ndarray(shape=(3,), dtype=np.float32))
        #graph_nx.add_node(i, objvel=np.ndarray(shape=(3,), dtype=np.float32))

    """ adding edges and features, the container does not have edges due to missing objpos """
    edge_tuples = [(a,b) for a, b in product(range(1, n_total_objects), range(1, n_total_objects)) if a != b]
    for edge in edge_tuples:
        graph_nx.add_edge(*edge, features=np.ndarray(shape=(3,), dtype=np.float32))

    """ adding global features """
    # 120*160*3(rgb)+120*160(seg)+120*160*3(depth)+1(timestep)+1(gravity)
    graph_nx.graph["features"] = np.zeros(shape=(134402,), dtype=np.float32)

    return graph_nx


def graph_to_input_and_targets_single_experiment(graph, features):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
      graph: An `nx.DiGraph` instance.

    Returns:
      The input `nx.DiGraph` instance.
      The target `nx.DiGraph` instance.

    Raises:
      ValueError: unknown node type
    """
    experiment_length = features['experiment_length']
    target_graphs = [graph.copy() for _ in range(experiment_length)]
    input_graphs = [graph.copy() for _ in range(experiment_length)]

    def create_node_feature(attr, features, step):
        if attr['type_name'] == 'container':
            """ container only has object segmentations """
            # pad up to fixed size since sonnet can only handle fixed-sized features
            res = attr['features']
            feature = features['object_segments'][0].flatten()
            res[:feature.shape[0]] = feature
            return res

        elif attr['type_name'] == 'gripper':
            """ gripper only has obj segs and gripper pos """
            obj_seg = features['object_segments'][1].flatten()
            pos = features['gripperpos'][step].flatten()
            # pad up to fixed size since sonnet can only handle fixed-sized features
            res = attr['features']
            concat_feature = np.concatenate((obj_seg, pos))
            res[:concat_feature.shape[0]] = concat_feature
            return res

        elif "manipulable" in attr['type_name']:
            obj_id = int(attr['type_name'].split("_")[2])
            obj_seg = features['object_segments'][obj_id].flatten()
            pos = features['objpos'][step][obj_id].flatten()
            vel = features['objvel'][step][obj_id].flatten() # todo: normalize by fps 1/30
            return np.concatenate((obj_seg, vel, pos))


    def create_edge_feature(receiver, sender, target_graph_i):
        node_feature_rcv = target_graph_i.nodes(data=True)[receiver]
        node_feature_snd = target_graph_i.nodes(data=True)[sender]
        pos1 = node_feature_rcv['features'][-3:]
        pos2 = node_feature_snd['features'][-3:]
        return np.absolute(pos1-pos2)

    for step in range(experiment_length):
        for node_index, node_feature in graph.nodes(data=True):
            target_graphs[step].add_node(node_index, features=create_node_feature(node_feature, features, step))

        """ add globals (image, segmentation, depth, gravity, time_step) """
        target_graphs[step].graph["features"] = np.concatenate((features['img'][step].flatten(), features['seg'][step].flatten(),
                                                          features['depth'][step].flatten(), np.atleast_1d(step+1), np.atleast_1d(
                                                            constants.g)))

    """ compute distances between every manipulable object (and gripper)"""
    for step in range(experiment_length):
        for receiver, sender, edge_feature in target_graphs[step].edges(data=True):
            target_graphs[step].add_edge(sender, receiver, features=create_edge_feature(receiver, sender, target_graphs[step]))

    input_graphs[0] = target_graphs[0].copy()


    return input_graphs, target_graphs


def get_graph_tuple(graph_nx):
    if type(graph_nx) is not list:
        graph_nx = [graph_nx]
    return utils_np.networkxs_to_graphs_tuple(graph_nx)

def get_graph_dict(graph_tuple):
    return utils_np.graphs_tuple_to_data_dicts(graph_tuple)

def get_graph_ph(graph_dicts):
    return utils_tf.placeholders_from_data_dicts(graph_dicts)

def print_graph_with_node_labels(graph_nx, label_keyword='features'):
    labels = nx.get_node_attributes(graph_nx, label_keyword)
    plt.figure(1, figsize=(11, 11))
    nx.draw(graph_nx, labels=labels, node_size=1000, font_size=15)
    plt.show()


def create_graph_and_get_graph_ph(config, n_total_objects, n_manipulable_objects):
    graph_nx = generate_singulation_graph(config, n_total_objects, n_manipulable_objects)
    graph_tuple = get_graph_tuple(graph_nx)
    graph_dict = get_graph_dict(graph_tuple)
    return get_graph_ph(graph_dict)


def create_singulation_graphs(config, batch_data, train_batch_size):
    input_graphs_all_experiments = []
    target_graphs_all_experiments = []
    graphs = []

    for i in range(train_batch_size):
        n_total_objects = batch_data[i]['n_total_objects']
        n_manipulable_objects = batch_data[i]['n_manipulable_objects']

        graph = generate_singulation_graph(config, n_total_objects, n_manipulable_objects)
        input_graphs, target_graphs = graph_to_input_and_targets_single_experiment(graph, batch_data[i])
        input_graphs_all_experiments.append(input_graphs)
        target_graphs_all_experiments.append(target_graphs)
        graphs.append(graph)

    return input_graphs_all_experiments, target_graphs_all_experiments, graphs


def create_placeholders(config, batch_data, batch_size):
    input_graphs, target_graphs, _ = create_singulation_graphs(config, batch_data, batch_size)
    input_phs = [utils_tf.placeholders_from_networkxs(ig, force_dynamic_num_graphs=True) for ig in input_graphs]
    target_phs = [utils_tf.placeholders_from_networkxs(tg, force_dynamic_num_graphs=True) for tg in target_graphs]

    return input_phs, target_phs, input_graphs, target_graphs


def create_feed_dict(config, input_ph, target_ph, input_graphs, target_graphs):
    input_tuple = utils_np.networkxs_to_graphs_tuple(input_graphs)
    target_tuple = utils_np.networkxs_to_graphs_tuple(target_graphs)

    return {input_ph: input_tuple, target_ph: target_tuple}

def create_feed_dict2(config, input_ph, target_ph, input_graphs, target_graphs):
    input_tuple = utils_np.networkxs_to_graphs_tuple(input_graphs)
    target_tuple = utils_np.networkxs_to_graphs_tuple(target_graphs)

    return {input_ph: input_tuple, target_ph: target_tuple}

