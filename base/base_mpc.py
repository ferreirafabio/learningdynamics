import tensorflow as tf
import scipy as sp
import numpy as np


class MPC:
    def __init__(self, sess, forward_model, train_data, test_data, config, logger, N=10):
        self.forward_model = forward_model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.test_data = test_data
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.N = N

    def tf_eval(self):
        raise NotImplementedError

    def solve(self, state):
        raise NotImplementedError



