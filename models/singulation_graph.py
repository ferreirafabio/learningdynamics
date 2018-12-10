import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.constants as constants
from itertools import product
from graph_nets import utils_tf, utils_np


def generate_singulation_graph(config, n_manipulable_objects):
    gripper_as_global = config.gripper_as_global

    graph_nx = nx.DiGraph()

    if not gripper_as_global:
        offset = 1
    else:
        offset = 0

    """ adding features to nodes """
    # 120*160*4(seg+rgb)
    #graph_nx.add_node(0, type_name='container')
    #graph_nx.add_node(0, features=np.zeros(shape=(config.node_output_size,), dtype=np.float32))

    if not gripper_as_global:
        """ if gripper should be a global attribute, include its data in global features """
        # 120*160*4(seg+rgb)+3(pos)
        graph_nx.add_node(0, type_name='gripper')
        graph_nx.add_node(0, features=np.zeros(shape=(config.node_output_size,), dtype=np.float32))

    for i in range(offset, n_manipulable_objects):
        ''' no multiple features -> flatten and concatenate everything '''
        # 120*160*4(seg+rgb)+3(pos)+3(vel)
        object_str = str(i)
        graph_nx.add_node(i, type_name='manipulable_object_' + object_str)
        graph_nx.add_node(i, features=np.zeros(shape=(config.node_output_size,), dtype=np.float32))

    """ adding edges and features, the container does not have edges due to missing objpos """
    edge_tuples = [(a,b) for a, b in product(range(0, graph_nx.number_of_nodes()), range(0, graph_nx.number_of_nodes())) if a != b]

    for edge in edge_tuples:
        graph_nx.add_edge(*edge, features=np.zeros(shape=(config.edge_output_size,), dtype=np.float32))

    """ adding global features """
    # 120*160*3(rgb)+120*160(seg)+120*160*3(depth)+1(timestep)+1(gravity)
    graph_nx.graph["features"] = np.zeros(shape=(config.global_output_size,), dtype=np.float32)

    return graph_nx


def graph_to_input_and_targets_single_experiment(config, graph, features, initial_pos_vel_known):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
      graph: An `nx.DiGraph` instance.

    Returns:
      The input `nx.DiGraph` instance.
      The target `nx.DiGraph` instance.

    Raises:
      ValueError: unknown node type
    """
    gripper_as_global = config.gripper_as_global
    data_offset_manipulable_objects = config.data_offset_manipulable_objects
    experiment_length = features['experiment_length']
    target_graphs = [graph.copy() for _ in range(experiment_length)]

    def create_node_feature(attr, features, step, use_object_seg_data_only_for_init):
        if attr['type_name'] == 'container':
            """ container only has object segmentations """
            # pad up to fixed size since sonnet can only handle fixed-sized features
            res = attr['features']
            if use_object_seg_data_only_for_init:
                feature = features['object_segments'][0].flatten()
            else:
                feature = features['object_segments'][step][0].flatten()
            res[:feature.shape[0]] = feature
            return res

        elif attr['type_name'] == 'gripper':
            """ gripper only has obj segs and gripper pos """
            if use_object_seg_data_only_for_init:
                obj_seg = features['object_segments'][1].flatten()
            else:
                obj_seg = features['object_segments'][step][1].flatten()
            pos = features['gripperpos'][step].flatten()
            # pad up to fixed size since sonnet can only handle fixed-sized features
            res = attr['features']
            res[:obj_seg.shape[0]] = obj_seg
            res[-3:] = pos
            return res

        elif "manipulable" in attr['type_name']:
            obj_id = int(attr['type_name'].split("_")[2])
            obj_id_segs = obj_id + data_offset_manipulable_objects
            # obj_seg will have data as following: (rgb, seg, optionally: depth)
            if use_object_seg_data_only_for_init:
                """ in this case, the nodes will have static visual information over time """
                obj_seg = features['object_segments'][obj_id].flatten()
            else:
                """ in this case, the nodes will have dynamic visual information over time """
                obj_seg = features['object_segments'][step][obj_id_segs].astype(np.float32).flatten()
            pos = features['objpos'][step][obj_id].flatten().astype(np.float32)
            vel = features['objvel'][step][obj_id].flatten().astype(np.float32) # todo: normalize by fps 1/30
            return np.concatenate((obj_seg, vel, pos))


    def create_edge_feature(receiver, sender, target_graph_i):
        node_feature_rcv = target_graph_i.nodes(data=True)[receiver]
        node_feature_snd = target_graph_i.nodes(data=True)[sender]
        """ the position is always located as the last three elements of the flattened feature vector """
        pos1 = node_feature_rcv['features'][-3:]
        pos2 = node_feature_snd['features'][-3:]
        return (pos1-pos2).astype(np.float32)

    for step in range(experiment_length):
        for node_index, node_feature in graph.nodes(data=True):
            node_feature = create_node_feature(node_feature, features, step, config.use_object_seg_data_only_for_init)
            target_graphs[step].add_node(node_index, features=node_feature)

        """ if gripper_as_global = True, graphs will have one node less
         add globals (image, segmentation, depth, gravity, time_step) """
        if gripper_as_global:
            target_graphs[step].graph["features"] = np.concatenate((features['img'][step].flatten(), features['seg'][step].flatten(),
                                                              features['depth'][step].flatten(), np.atleast_1d(step+1), np.atleast_1d(
                                                                constants.g), features['gripperpos'][step].flatten())).astype(np.float32)
        else:
            target_graphs[step].graph["features"] = np.concatenate((features['img'][step].flatten(), features['seg'][step].flatten(),
                                                              features['depth'][step].flatten(), np.atleast_1d(step+1), np.atleast_1d(
                                                                constants.g))).astype(np.float32)

    """ compute distances between every manipulable object (and gripper if not gripper_as_global) """
    for step in range(experiment_length):
        for receiver, sender, edge_feature in target_graphs[step].edges(data=True):
            edge_feature = create_edge_feature(receiver, sender, target_graphs[step])
            target_graphs[step].add_edge(sender, receiver, features=edge_feature)

    input_graph = target_graphs[0].copy()

    # todo: following code assumes all nodes are of type 'manipulable'
    """ set velocity and position info to zero """
    if not initial_pos_vel_known:
        """ for all nodes """
        for idx, node_feature in input_graph.nodes(data=True):
            feat = node_feature['features']
            feat[-6:] = 0
            input_graph.add_node(idx, features=feat)
        """ for all edges """
        for receiver, sender, edge_feature in input_graph.edges(data=True):
            feat = edge_feature['features']
            feat[:] = 0
            input_graph.add_edge(sender, receiver, features=feat)

    return input_graph, target_graphs


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


def create_singulation_graphs(config, batch_data, train_batch_size, initial_pos_vel_known):
    input_graphs_all_experiments = []
    target_graphs_all_experiments = []
    graphs = []

    for i in range(train_batch_size):
        n_manipulable_objects = batch_data[i]['n_manipulable_objects']

        graph = generate_singulation_graph(config, n_manipulable_objects)
        input_graphs, target_graphs = graph_to_input_and_targets_single_experiment(config, graph, batch_data[i], initial_pos_vel_known)
        input_graphs_all_experiments.append(input_graphs)
        target_graphs_all_experiments.append(target_graphs)
        graphs.append(graph)

    return input_graphs_all_experiments, target_graphs_all_experiments, graphs


def create_graphs(config, batch_data, batch_size, initial_pos_vel_known):
    input_graphs, target_graphs, _ = create_singulation_graphs(config, batch_data, batch_size, initial_pos_vel_known=initial_pos_vel_known)

    if not initial_pos_vel_known:
        """ sanity checking one of the graphs """
        for _, node_feature in input_graphs[1].nodes(data=True):
            assert not np.any(node_feature['features'][-6:])

        for _, _, edge_feature in input_graphs[1].edges(data=True):
            assert not np.any(edge_feature['features'][-3:])

    return input_graphs, target_graphs


def create_placeholders(config, batch_data):
    input_graphs, target_graphs, _ = create_singulation_graphs(config, batch_data, train_batch_size=1, initial_pos_vel_known=False)

    input_ph = utils_tf.placeholders_from_networkxs(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs[0], force_dynamic_num_graphs=True)
    return input_ph, target_ph


def create_feed_dict(input_ph, target_ph, input_graphs, target_graphs):
    input_tuple = utils_np.networkxs_to_graphs_tuple([input_graphs])
    target_tuple = utils_np.networkxs_to_graphs_tuple(target_graphs)

    return {input_ph: input_tuple, target_ph: target_tuple}