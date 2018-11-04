import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import sonnet as snt
import tensorflow as tf
from graph_nets import utils_tf, utils_np


class SingulationGraph():
    def __init__(self, config, n_total_objects, n_manipulable_objects):
        self.config = config
        self.n_nodes_attr = self.config["n_nodes_attr"]
        #self.n_edges_attr = self.config["n_edges_attr"]
        #self.n_globals_attr = self.config["n_globals_attr"]
        #self.global_output_size = self.config["global_output_size"]

        self.G = nx.complete_graph(n_total_objects, nx.MultiDiGraph)
        # todo: check whether node features can have different shapes

        """ adding features to nodes """
        # 120*160*4(seg+rgb)
        self.G.node[0]['type'] = 'container'
        self.G.node[0]['features'] = np.ndarray(shape=(76800,), dtype=np.float32)

        # 120*160*4(seg+rgb)+3(pos)
        self.G.node[1]['type'] = 'gripper'
        self.G.node[1]['features'] = np.ndarray(shape=(76803,), dtype=np.float32)


        for i in range(2, n_manipulable_objects+2):
            ''' no multiple features -> flatten and concatenate everything '''
            # 120*160*4(seg+rgb)+3(pos)+3(vel)
            self.G.node[i]['type'] = 'manipulable_object'
            self.G.node[i]['features'] = np.ndarray(shape=(76806,), dtype=np.float32)

        """ adding features to edges """
        # todo


        self.graphs_tuple = utils_np.networkxs_to_graphs_tuple([self.G])

    def build_graph(self):
        return utils_tf.data_dicts_to_graphs_tuple([self.data_dict])



