import tensorflow as tf
import numpy as np


# placeholders
ph_manipulable_node = {
    'seg_img': tf.placeholder(tf.int8, shape=[None, 120, 160, 4]), # contains rgb and segmentation channels
    'pos': tf.placeholder(tf.float64, shape=[None, 3]),
    'vel': tf.placeholder(tf.float64, shape=[None, 3])
}

ph_gripper_node = {
    'seg_img': tf.placeholder(tf.int8, shape=[None, 120, 160, 4]), # contains rgb and segmentation channels
    'gripperpos': tf.placeholder(tf.float64, shape=[None, 3])
}

ph_container_node = {
    'seg_img': tf.placeholder(tf.int8, shape=[None, 120, 160, 4]) # contains rgb and segmentation channels
}

def edge(graph_instance, nodeA, nodeB):

    raise NotImplementedError

def add_node(graph_instance, node_tensor):

    raise NotImplementedError




if __name__ == '__main__':
    edge()