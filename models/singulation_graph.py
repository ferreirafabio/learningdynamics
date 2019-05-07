import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.constants as constants
from utils.utils import make_all_runnable_in_session
from itertools import product
from graph_nets import utils_tf, utils_np
from utils.utils import chunks, get_correct_image_shape
import random


def generate_singulation_graph(config, n_manipulable_objects):
    gripper_as_global = config.gripper_as_global

    #graph_nx = nx.DiGraph()
    graph_nx = nx.OrderedMultiDiGraph()

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
        # 120*160*4(seg+rgb)+3(vel)+3(pos)
        object_str = str(i)
        graph_nx.add_node(i, type_name='manipulable_object_' + object_str)
        graph_nx.add_node(i, features=np.zeros(shape=(config.node_output_size,), dtype=np.float32))

    """ adding edges and features, the container does not have edges due to missing objpos """
    edge_tuples = [(a,b) for a, b in product(range(0, graph_nx.number_of_nodes()), range(0, graph_nx.number_of_nodes())) if a != b]

    for edge in edge_tuples:
        graph_nx.add_edge(*edge, key=0, features=np.zeros(shape=(config.edge_output_size,), dtype=np.float32))

    """ adding global features """
    # 120*160*3(rgb)+120*160(seg)+120*160*3(depth)+1(timestep)+1(gravity)
    graph_nx.graph["features"] = np.zeros(shape=(config.global_output_size,), dtype=np.float32)

    return graph_nx


def graph_to_input_and_targets_single_experiment(config, graph, features, initial_pos_vel_known, return_only_unpadded=False):
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

    """ handles the testing cycles when a different number of rollouts shall be predicted than seen in training """
    if config.n_rollouts is not experiment_length:
        experiment_length = config.n_rollouts

    target_graphs = [graph.copy() for _ in range(experiment_length)]

    def create_node_feature(attr, features, step, config):
        if attr['type_name'] == 'container':
            """ container only has object segmentations """
            # pad up to fixed size since sonnet can only handle fixed-sized features
            res = attr['features']
            if config.use_object_seg_data_only_for_init:
                feature = features['object_segments'][1].flatten()
            else:
                feature = features['object_segments'][step][1].flatten()
            res[:feature.shape[0]] = feature
            return res

        elif attr['type_name'] == 'gripper':
            """ gripper only has obj segs and gripper pos """
            if config.use_object_seg_data_only_for_init:
                obj_seg = features['object_segments'][0].flatten()
            else:
                obj_seg = features['object_segments'][step][0].flatten()
            pos = features['gripperpos'][step].flatten().astype(np.float32)
            vel = features['grippervel'][step].flatten().astype(np.float32)
            return np.concatenate((obj_seg, vel, pos))

        elif "manipulable" in attr['type_name']:
            """ we assume shape (image features, vel(3dim), pos(3dim)) """
            obj_id = int(attr['type_name'].split("_")[2])
            obj_id_segs = obj_id + data_offset_manipulable_objects

            # obj_seg will have data as following: (rgb, seg, optionally: depth)
            if config.use_object_seg_data_only_for_init:
                """ in this case, the nodes will have static visual information over time """
                obj_seg = features['object_segments'][obj_id].flatten()
            else:
                """ in this case, the nodes will have dynamic visual information over time """
                obj_seg = features['object_segments'][step][obj_id_segs].astype(np.float32)
                """ nodes have full access to scene observation (i.e. rgb and depth) """
                if config.nodes_get_full_rgb_depth:
                    rgb = features["img"][step].astype(np.float32)
                    depth = features["depth"][step].astype(np.float32)
                    obj_seg[:,:,:3] = rgb
                    obj_seg[:,:,-3:] = depth

                obj_seg = obj_seg.flatten()
            pos = features['objpos'][step][obj_id].flatten().astype(np.float32)

            # normalize velocity
            # """ (normalized) velocity is computed here since rolled indexing in
            # tfrecords seems not straightforward """
            # if step == 0:
            #    diff = np.zeros(shape=3, dtype=np.float32)
            # else:
            #    diff = features['objpos'][step-1][obj_id] - features['objpos'][step][obj_id]
            #    if config.normalize_data:
            #        vel = normalize_list([diff])[0]
            #vel = (diff * 240.0).flatten().astype(np.float32)
            vel = features['objvel'][step][obj_id].flatten().astype(np.float32)
            if config.remove_pos_vel:
                pos = np.zeros(shape=np.shape(pos), dtype=np.float32)
                vel = np.zeros(shape=np.shape(vel), dtype=np.float32)
            return np.concatenate((obj_seg, vel, pos))

    def create_edge_feature_distance(receiver, sender, target_graph_i):
        node_feature_rcv = target_graph_i.nodes(data=True)[receiver]
        node_feature_snd = target_graph_i.nodes(data=True)[sender]
        """ the position is always the last three elements of the flattened feature vector """
        pos1 = node_feature_rcv['features'][-3:]
        pos2 = node_feature_snd['features'][-3:]
        return (pos1-pos2).astype(np.float32)

    def create_edge_feature(sender, target_graph, target_graph_previous, seg_as_edges, img_shape=None):
        if not seg_as_edges:
            node_feature_snd_prev = target_graph_previous.nodes(data=True)[sender]
            node_feature_snd = target_graph.nodes(data=True)[sender]
            """ the position is always the last three elements of the flattened feature vector """
            pos_prev = node_feature_snd_prev["features"][-3:]
            vel_pos = node_feature_snd['features'][-6:]
            vel_pos = np.insert(vel_pos, 3, pos_prev)
            """ will yield (vel_t, pos_{t-1}, pos_t)"""
            return vel_pos.astype(np.float32)
        else:
            node_feature = target_graph.nodes(data=True)[sender]['features'][:-6]
            node_feature = np.reshape(node_feature, img_shape)
            return node_feature[:,:,3].flatten()

    input_control_graphs = []

    for step in range(experiment_length):

        for node_index, node_feature in graph.nodes(data=True):
            node_feature = create_node_feature(node_feature, features, step, config)
            target_graphs[step].add_node(node_index, features=node_feature)

        """ if gripper_as_global = True, graphs will have one node less
         add globals (image, segmentation, depth, gravity, time_step) """
        if gripper_as_global:
            if config.global_output_size == 5:
                global_features = np.concatenate((np.atleast_1d(step),
                                                 np.atleast_1d(constants.g),
                                                 features['gripperpos'][step].flatten()
                                                  )).astype(np.float32)
            elif config.global_output_size == 9:
                padding_flag = 1 if step >= features["unpadded_experiment_length"] else 0
                global_features = np.concatenate((np.atleast_1d(padding_flag),
                                                  np.atleast_1d(step),
                                                  np.atleast_1d(constants.g),
                                                  features['gripperpos'][step].flatten(),
                                                  features['grippervel'][step].flatten(),
                                                  )).astype(np.float32)
            else:
                global_features = np.concatenate((features['img'][step].flatten(),
                                    features['seg'][step].flatten(),
                                    features['depth'][step].flatten(),
                                    np.atleast_1d(step),
                                    np.atleast_1d(constants.g),
                                    features['gripperpos'][step].flatten())
                                   ).astype(np.float32)


            target_graphs[step].graph["features"] = global_features

            """ assign gripperpos to input control graphs """
            input_control_graph = graph.copy()
            for i in range(input_control_graph.number_of_nodes()):
                input_control_graph.nodes(data=True)[i]["features"] = None
            for receiver, sender, edge_feature in input_control_graph.edges(data=True):
                input_control_graph[sender][receiver][0]['features'] = None

            input_control_graph.graph["features"] = global_features

            assert target_graphs[step].graph["features"].shape[0] == config.global_output_size
            assert input_control_graph.graph["features"].shape[0] == config.global_output_size
            input_control_graphs.append(input_control_graph)

        else:
            if config.global_output_size == 2:
                target_graphs[step].graph["features"] = np.concatenate((np.atleast_1d(step),
                                                                        np.atleast_1d( constants.g)
                                                                        )).astype(np.float32)

            #assert target_graphs[step].graph["features"].shape[0]-3 == config.global_output_size
            input_control_graphs = None

    """ compute distances between every manipulable object (and gripper if not gripper_as_global) """
    for step in range(experiment_length):
        for sender, receiver, edge_feature in target_graphs[step].edges(data="features"):
            if step == 0:
                target_graphs_previous = target_graphs[step]
            else:
                target_graphs_previous = target_graphs[step-1]
            edge_feature = create_edge_feature(sender=sender, target_graph=target_graphs[step],
                                                       target_graph_previous=target_graphs_previous,
                                                       seg_as_edges=config.edges_carry_segmentation_data,
                                                       img_shape=get_correct_image_shape(config, get_type='all'))
            if config.remove_edges:
                edge_feature = np.zeros(shape=np.shape(edge_feature), dtype=np.float32)

            target_graphs[step].add_edge(sender, receiver, key=0, features=edge_feature)

    input_graphs = []
    for i in range(experiment_length-1):
        inp = target_graphs[i].copy()
        inp.graph["features"] = input_control_graphs[i+1].graph["features"]
        input_graphs.append(inp)

    #input_graph = target_graphs[0].copy()
    target_graphs = target_graphs[1:]  # first state is used for init
    #if gripper_as_global:
    #    """ gripperpos and grippervel always reflect the current step. However, we are interested in predicting
    #    the effects of a new/next control command --> shift by one """
    #    input_control_graphs = input_control_graphs[1:]

    # todo: following code assumes all nodes are of type 'manipulable'
    """ set velocity and position info to zero """
    if not initial_pos_vel_known:
        """ for all nodes """
        for graph in input_graphs:
            for idx, node_feature in graph.nodes(data=True):
                feat = node_feature['features']
                feat[-6:] = 0
                graph.add_node(idx, features=feat)
                """ for all edges """
            for receiver, sender, edge_feature in graph.edges(data=True):
                feat = edge_feature['features']
                feat[:] = 0
                graph.add_edge(sender, receiver, features=feat)

    if return_only_unpadded:
        input_graphs = [graph for graph in input_graphs if graph.graph['features'][0] == 0]
        target_graphs = [graph for graph in target_graphs if graph.graph['features'][0] == 0]

    """ check if the gripper pos+vel in the input graph are values from the next time step """
    assert (input_graphs[0].graph['features'] == target_graphs[0].graph['features']).all()

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
    plt.figure(figsize=(11, 11))
    nx.draw(graph_nx, labels=labels, node_size=1000, font_size=15)
    plt.show()

def print_graph_with_node_and_edge_labels(graph_nx, label_keyword="features"):
    labels1 = nx.get_node_attributes(graph_nx, "features")
    labels2 = nx.get_node_attributes(graph_nx, "type_name")
    edge_labels = nx.get_edge_attributes(graph_nx,'features')
    edge_labels = {k[:2]:v for k,v in edge_labels.items()}

    plt.figure(figsize=(11, 11))
    pos = nx.spring_layout(graph_nx)
    nx.draw_networkx_labels(graph_nx, pos, labels=labels1)
    nx.draw(graph_nx, labels=labels2, node_size=1000, font_size=10, label_position=0.3)
    nx.draw_networkx_edge_labels(graph_nx, pos, edge_labels=edge_labels)

    plt.show()


def create_graph_batch(config, graph, batch_data, initial_pos_vel_known, shuffle=True, return_only_unpadded=True):
    input_graph_lst, target_graph_lst = [], []
    for data in batch_data:
        input_graphs, target_graphs = graph_to_input_and_targets_single_experiment(config, graph, data, initial_pos_vel_known, return_only_unpadded=return_only_unpadded)
        if not shuffle:
            input_graph_lst.append(input_graphs)
            target_graph_lst.append(target_graphs)
        else:
            input_graph_lst.append((input_graphs, data['experiment_id']))
            target_graph_lst.append((target_graphs, data['experiment_id']))

    if not shuffle:
        input_graph_lst = list(input_graph_lst)
        target_graph_lst = list(target_graph_lst)
        input_graph_lst = list(chunks(input_graph_lst, config.train_batch_size))
        target_graph_lst = list(chunks(target_graph_lst, config.train_batch_size))

        return input_graph_lst, target_graph_lst


    "flatten lists"
    input_graph_lst = [(lst, tpl_e2) for tpl_e1, tpl_e2 in input_graph_lst for lst in tpl_e1]
    target_graph_lst = [(lst, tpl_e2) for tpl_e1, tpl_e2 in target_graph_lst for lst in tpl_e1]

    "shuffle lists"
    shuffled_list = list(zip(input_graph_lst, target_graph_lst))

    random.shuffle(shuffled_list)

    """ ensure that no batch has input/output graph with the same experiment id """
    lst = []
    sublist = []
    while True:
        smpl = random.choice(shuffled_list)
        if smpl is None or len(lst) > 1:
            break
        ids_in_list = [elements[0][1] for elements in sublist]
        if smpl[0][1] not in ids_in_list:
            sublist.append(smpl)
            shuffled_list.remove(smpl)
        if len(sublist) == config.train_batch_size:
            lst.append(sublist)
            sublist = []

    input_batches = []
    target_batches = []
    for batch in lst:
        input_batch = []
        target_batch = []
        for sublist in batch:
            input_batch.append(sublist[0][0])
            target_batch.append(sublist[1][0])

        input_batches.append(input_batch)
        target_batches.append(target_batch)

    return input_batches, target_batches



def create_singulation_graphs(config, batch_data, initial_pos_vel_known, batch_processing=True, shuffle=True, return_only_unpadded=True):
    if not batch_processing:
        n_manipulable_objects = batch_data['n_manipulable_objects']
        graph = generate_singulation_graph(config, n_manipulable_objects)
        input_graphs, target_graphs = graph_to_input_and_targets_single_experiment(config, graph, batch_data, initial_pos_vel_known, return_only_unpadded=return_only_unpadded)
    else:
        n_manipulable_objects = batch_data[0]['n_manipulable_objects']
        graph = generate_singulation_graph(config, n_manipulable_objects)
        input_graphs, target_graphs = create_graph_batch(config, graph, batch_data, initial_pos_vel_known, shuffle=shuffle, return_only_unpadded=return_only_unpadded)

    return input_graphs, target_graphs


def create_graphs(config, batch_data, initial_pos_vel_known, batch_processing=True, shuffle=True, return_only_unpadded=True):
    input_graphs, target_graphs = create_singulation_graphs(config, batch_data, initial_pos_vel_known=initial_pos_vel_known, batch_processing=batch_processing, shuffle=shuffle, return_only_unpadded=return_only_unpadded)

    if not initial_pos_vel_known:
        _sanity_check_pos_vel(input_graphs)

    return input_graphs, target_graphs


def create_placeholders(config, batch_data, batch_processing=True):
    """ if gripper_as_global = False, this function will still return 3 values (input_ph, target_ph, input_ctrl_ph) but the last will
    (input_ctrl_ph) will be None, caller needs to check this """
    input_graphs, target_graphs = create_singulation_graphs(config, batch_data, initial_pos_vel_known=config.initial_pos_vel_known, batch_processing=batch_processing)

    if batch_processing:
        """ in this case we get a list of chunk lists (each representing a full batch)"""
        input_graphs = input_graphs[0]
        target_graphs = target_graphs[0]
    else:
        input_graphs = [input_graphs[0]]
        target_graphs = [target_graphs[0]]

    input_ph = utils_tf.placeholders_from_networkxs(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs, force_dynamic_num_graphs=True)

    return input_ph, target_ph


def create_feed_dict(input_ph, target_ph, input_graphs, target_graphs, batch_processing=True):
    if batch_processing:
        input_graphs = input_graphs
        target_graphs = target_graphs
    else:
        input_graphs = [input_graphs]
        target_graphs = [target_graphs]

    input_tuple = utils_np.networkxs_to_graphs_tuple(input_graphs)
    target_tuple = utils_np.networkxs_to_graphs_tuple(target_graphs)

    input_dct = utils_tf.get_feed_dict(input_ph, input_tuple)
    target_dct = utils_tf.get_feed_dict(target_ph, target_tuple)

    input_ph_runnable, target_ph_runnable = make_all_runnable_in_session(input_ph, target_ph)

    return input_ph_runnable, target_ph_runnable, {**input_dct, **target_dct}


def _sanity_check_pos_vel(input_graphs):
    """ asserts whether position and velocity are zero """
    """ sanity checking one of the graphs """
    for _, node_feature in input_graphs[1].nodes(data=True):
        assert not np.any(node_feature['features'][-6:])

    for _, _, edge_feature in input_graphs[1].edges(data=True):
        assert not np.any(edge_feature['features'][-3:])


def networkx_graphs_to_images(config, input_graphs_batch, target_graphs_batch):
    in_image = []
    gt_label = []
    in_segxyz = []
    in_control = []

    for graph in input_graphs_batch[0]:
        for i, node_feature in graph.nodes(data=True):
            in_control.append(graph.graph['features'][3:6:])  # control input for the next time step, its a 9-tuple (step,g,posx,posy,posz,velx,vely,velz)

            node_feature = node_feature['features'][:-6]
            node_feature_reshaped = np.reshape(node_feature, [120, 160, 7])
            seg_of_node_i = node_feature_reshaped[:,:,3]

            """ we keep full rgb and depth, if masked, multiply by seg """
            xyz = node_feature_reshaped[:, :, -3:]
            seg = np.expand_dims(seg_of_node_i, axis=-1)
            seg_xyz = np.concatenate([seg, xyz], axis=-1)

            rgb = node_feature_reshaped[:, :, :3]
            in_image.append(rgb)
            in_segxyz.append(seg_xyz)

    for graph in target_graphs_batch[0]:
        for i, node_feature in graph.nodes(data=True):
            node_feature = node_feature['features'][:-6]
            node_feature_reshaped = np.reshape(node_feature, [120, 160, 7])
            seg_of_node_i = node_feature_reshaped[:, :, 3]

            gt_label.append(seg_of_node_i)

    #import matplotlib
    #matplotlib.use("TkAgg")

    #import matplotlib.pyplot as plt

    #plt.figure("input")
    #plt.imshow(in_segxyz[12][:, :, 0])

    #plt.figure("target")
    #plt.imshow(gt_label[12])

    #plt.figure("in_image")
    #plt.imshow(in_image[12])
    #plt.show()

    in_segxyz = np.array(in_segxyz)
    in_image = np.array(in_image)
    in_control = np.array(in_control)
    #print("converting gt_label to bool array")
    #gt_label = np.array(gt_label, dtype=np.bool)
    gt_label = np.array(gt_label)


    return in_segxyz, in_image, in_control, gt_label

