import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import sonnet as snt
import tensorflow as tf
from graph_nets import utils_tf, utils_np


class SingulationGraph():
    def __init__(self, config, node_output_size):
        self.config = config
        self.n_nodes_attr = self.config["n_nodes_attr"]
        #self.n_edges_attr = self.config["n_edges_attr"]
        #self.n_globals_attr = self.config["n_globals_attr"]
        #self.global_output_size = self.config["global_output_size"]

        self.G = nx.MultiDiGraph()

        """ adding nodes and features """
        for i in range(node_output_size):
            self.G.add_node(i, features=tf.placeholder(tf.uint8, shape=(120, 160, 4)))
            ''' can we add multiple features of unequal shape to the same node???'''
            self.G.add_node(i, features=tf.placeholder(tf.uint8, shape=(120, 160, 4)))
        self.G.add_nodes_from(np.arange(node_output_size))


        self.nx_graph = [nx.complete_graph(node_output_size, nx.MultiDiGraph())]

        #self.nx_graph = nx.MultiDiGraph()
        #self.nx_graph.add_nodes_from(np.arange(node_output_size))


        #self.nodes = np.ndarray(shape=(node_output_size, self.n_nodes_attr), dtype=np.float32)
        #self.globals = np.ndarray(shape=(self.global_output_size, self.n_globals_attr), dtype=np.float32)
        #self.edges, self.senders, self.receivers = [], [], []

        self.graph = self.build_graph()

    def build_graph(self):

        #data_dict = {
        #    "globals": self.globals,
        #    "nodes": self.nodes,
        #    "edges": self.edges,
        #    "receivers": self.receivers,
        #    "senders": self.senders
        #}

        return utils_np.networkxs_to_graphs_tuple(self.nx_graph)


