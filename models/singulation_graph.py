import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from graph_nets import utils_tf, utils_np


def generate_singulation_graph(config, n_total_objects, n_manipulable_objects):

    n_nodes_attr = config["n_nodes_attr"]
    #self.n_edges_attr = self.config["n_edges_attr"]
    #self.n_globals_attr = self.config["n_globals_attr"]
    #self.global_output_size = self.config["global_output_size"]

    graph_nx = nx.DiGraph()

    """ adding features to nodes """
    # 120*160*4(seg+rgb)
    #graph_nx.add_node(0, type_name='container')
    graph_nx.add_node(0, features=np.ndarray(shape=(76806,), dtype=np.float32))

    # 120*160*4(seg+rgb)+3(pos)
    #graph_nx.add_node(1, type_name='gripper')
    graph_nx.add_node(1, features=np.ndarray(shape=(76806,), dtype=np.float32))


    for i in range(2, n_manipulable_objects+2):
        ''' no multiple features -> flatten and concatenate everything '''
        # 120*160*4(seg+rgb)+3(pos)+3(vel)
        #graph_nx.add_node(i, type_name='manipulable_object')
        graph_nx.add_node(i, features=np.ndarray(shape=(76806,), dtype=np.float32))

    """ adding edges and features """
    edge_tuples = [(a,b) for a, b in product(range(n_total_objects), range(n_total_objects)) if a != b]
    for edge in edge_tuples:
        graph_nx.add_edge(*edge, features=np.ndarray(shape=(3,), dtype=np.float32))

    """ adding global features """
    # 120*160*3(rgb)+1(timestep)+1(gravity)
    graph_nx.graph["features"] = np.ndarray(shape=(57602,), dtype=np.float32)

    return graph_nx


def graph_to_input_target(graph, features):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
      graph: An `nx.DiGraph` instance.

    Returns:
      The input `nx.DiGraph` instance.
      The target `nx.DiGraph` instance.

    Raises:
      ValueError: unknown node type
    """
    input_graph = graph.copy()
    target_graph = graph.copy()

    for node_index, node_feature in graph.nodes(data=True):
        # todo
        input_graph.add_node(node_index, features="a")
        target_graph.add_node(node_index, features="a")


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

def create_placeholders(config, features, n_graphs, n_total_objects, n_manipulable_objects):
    input_graphs = []
    target_graphs = []
    graph = generate_singulation_graph(config, n_total_objects, n_manipulable_objects)
    for _ in range(n_graphs):
        input_graph, target_graph = graph_to_input_target(graph, features)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)

    input_ph = utils_tf.placeholders_from_networkxs(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs, force_dynamic_num_graphs=True)
    return input_ph, target_ph
