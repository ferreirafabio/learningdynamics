import numpy as np
import networkx as nx
from itertools import combinations
import sonnet as snt
import tensorflow as tf
from graph_nets import utils_tf


class SingulationGraph():
    def __init__(self, config, node_output_size, data, name="Graph"):
        self.config = config
        self.n_nodes_attr = self.config["n_nodes_attr"]
        self.n_edges_attr = self.config["n_edges_attr"]
        self.n_globals_attr = self.config["n_globals_attr"]
        self.global_output_size = self.config["global_output_size"]

        self.nx_graph = nx.complete_graph(node_output_size, nx.MultiDiGraph())

        nose =

        self.nx_graph = nx.MultiDiGraph()
        self.nx_graph.add_nodes_from(np.arange(node_output_size))





        self.nodes = np.ndarray(shape=(node_output_size, self.n_nodes_attr), dtype=np.float32)
        self.globals = np.ndarray(shape=(self.global_output_size, self.n_globals_attr), dtype=np.float32)
        self.edges, self.senders, self.receivers = [], [], []


        self.graph = self.build_graph()

    def build_graph(self):
        data_dict = {
            "globals": self.globals,
            "nodes": self.nodes,
            "edges": self.edges,
            "receivers": self.receivers,
            "senders": self.senders
        }
  # Nodes
        return utils_tf.data_dicts_to_graphs_tuple(data_dict)


if __name__ == '__main__':
    graph = Graph(config=None, node_output_size=5, global_output_size=5)