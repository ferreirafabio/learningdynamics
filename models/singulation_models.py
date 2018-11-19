# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model architectures for the singulation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
from base.base_model import BaseModel
import sonnet as snt
import tensorflow as tf


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
                node_model_fn=EncodeProcessDecode.make_mlp_model,
                global_model_fn=EncodeProcessDecode.make_mlp_model)

    def _build(self, inputs):
        return self._network(inputs)


class CNNEncoderGraphIndependent(snt.AbstractModule):
    """GraphNetwork with CNN node and MLP edge / global models."""

    def __init__(self, name="CNNEncoderGraphIndependent"):
        super(CNNEncoderGraphIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
              edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
              node_model_fn=EncodeProcessDecode.make_cnn_model,
              global_model_fn=EncodeProcessDecode.make_mlp_model)

    def _build(self, inputs, **kwargs):
        return self._network(inputs, **kwargs)

class CNNDecoderGraphIndependent(snt.AbstractModule):
    """Graph decoder network with Transpose CNN node and MLP edge / global models."""

    def __init__(self, name="CNNDecoderGraphIndependent"):
        super(CNNDecoderGraphIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
                node_model_fn=EncodeProcessDecode.make_transpose_cnn_model,
                global_model_fn = EncodeProcessDecode.make_mlp_model)

    def _build(self, inputs, **kwargs):
        return self._network(inputs, **kwargs)


class MLPGraphNetwork(snt.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
          self._network = modules.GraphNetwork(EncodeProcessDecode.make_mlp_model_edges,
                                               EncodeProcessDecode.make_mlp_model,
                                               EncodeProcessDecode.make_mlp_model)

    def _build(self, inputs):
        return self._network(inputs)


class EncodeProcessDecode(snt.AbstractModule, BaseModel):
    """Full encode-process-decode model.

    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
    """
    def __init__(self, config, name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        EncodeProcessDecode.n_layers = config.n_layers
        EncodeProcessDecode.n_neurons = config.n_neurons
        EncodeProcessDecode.n_neurons_edges = config.n_neurons_edges
        EncodeProcessDecode.n_layers_edges = config.n_layers_edges
        EncodeProcessDecode.n_convnet1D_filters_per_layer = config.n_convnet1D_filters_per_layer
        EncodeProcessDecode.convnet1D_kernel_size = config.convnet1D_kernel_size


        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        # init the batch counter
        self.init_batch_step()

        self.use_cnn = self.config.use_cnn

        if self.use_cnn:
            self._encoder = CNNEncoderGraphIndependent()
            self._decoder = CNNDecoderGraphIndependent()
        else:
            self._encoder = MLPGraphIndependent()
            self._decoder = MLPGraphIndependent()

        self._core = MLPGraphNetwork()

        self.init_ops()

        self.node_output_size = config.node_output_size
        self.edge_output_size = config.edge_output_size
        self.global_output_size = config.global_output_size

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        #self.exp_length = tf.placeholder(tf.int32, shape=(), name='exp_length')
        #self.is_training = tf.placeholder(tf.bool, shape=())

        self.init_transform()




    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))

        return output_ops


    def _build2(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        index = tf.constant(0)
        output_ops = tf.Variable([])

        def condition(index, output_ops, latent):
            return tf.less(index, num_processing_steps)

        def body(index, output_ops, latent):
            core_input = utils_tf.concat([latent0, latent], axis=1) # todo: change to 1
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            d = tf.cast(self._output_transform(decoded_op), tf.float32)
            output_ops = utils_tf.concat([output_ops, d], 0)

            return tf.add(index, 1), output_ops, latent

        return tf.while_loop(condition, body, loop_vars=[index, output_ops, latent])


    def create_loss_ops(self, target_op, output_ops):
        """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
        is a list of graph tuples with shape (n_nodes, exp_len) --> split at the first dimension in order to compute node-wise MSE error
        --> same applies for edges """
        mult = tf.constant([len(output_ops)])
        n_nodes = [tf.shape(output_ops[0].nodes)[0]]
        n_edges = [tf.shape(output_ops[0].edges)[0]]

        node_splits = tf.split(target_op.nodes, num_or_size_splits=tf.tile(n_nodes, mult), axis=0)
        edge_splits = tf.split(target_op.edges, num_or_size_splits=tf.tile(n_edges, mult), axis=0)

        """ if object seg data is only used for init, the ground truth features in the rest of the sequence are static except position 
        --> in this case compute loss only over the position since image prediction is infeasible """
        if self.config.use_object_seg_data_only_for_init:
            loss_ops = [
                # compute loss of the nodes only over velocity and position and not over ground truth static images
                tf.losses.mean_squared_error(output_op.nodes, node_splits[i][:, -6:]) +
                tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
                    for i, output_op in enumerate(output_ops)
            ]
        else:
            loss_ops = [
                tf.losses.mean_squared_error(output_op.nodes, node_splits[i]) +
                tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
                for i, output_op in enumerate(output_ops)
            ]
        #    print(len(loss_ops))
        # todo: might use weighted MSE loss here
        # todo: perhaps include global attributes into loss function

        return tf.reduce_mean(loss_ops)


    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.cur_batch_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_batch_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.cur_batch_tensor = tf.Variable(0, trainable=False, name='cur_batch')
            self.increment_cur_batch_tensor = tf.assign(self.cur_batch_tensor, self.cur_batch_tensor + 1)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_checkpoints_to_keep)

    def init_ops(self):
        self.step_op = None
        self.loss_op_train = None
        self.loss_op_test = None

        self.loss_ops_train = None
        self.loss_ops_test = None

    def init_transform(self):
        # Transforms the outputs into the appropriate shapes.
        if self.edge_output_size is None:
          edge_fn = None
        else:
          edge_fn = lambda: snt.Linear(self.edge_output_size, name="edge_output")
        if self.node_output_size is None:
          node_fn = None
        else:
          node_fn = lambda: snt.Linear(self.node_output_size, name="node_output")
        if self.global_output_size is None:
          global_fn = None
        else:
          global_fn = lambda: snt.Linear(self.global_output_size, name="global_output")
        with self._enter_variable_scope():
          self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn)

    def is_encoder(self, boolean_value):
        return boolean_value

    @staticmethod
    def make_mlp_model():
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
          A Sonnet module which contains the MLP and LayerNorm.
        """
        return snt.Sequential([snt.nets.MLP([EncodeProcessDecode.n_neurons] * EncodeProcessDecode.n_layers, activate_final=True), snt.LayerNorm()])


    @staticmethod
    # since edges are very low-dim, use different number of neurons and layers
    def make_mlp_model_edges():
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
          A Sonnet module which contains the MLP and LayerNorm.
        """
        return snt.Sequential([snt.nets.MLP([EncodeProcessDecode.n_neurons_edges] * EncodeProcessDecode.n_layers_edges, activate_final=True), snt.LayerNorm()])


    @staticmethod
    def make_cnn_model():
        def convnet1d(inputs):
            # input shape is (batch_size, feature_length) but CNN operates on depth channels --> (batch_size, feature_length, 1)
            inputs = tf.expand_dims(inputs, axis=2)

            outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(inputs)
            outputs = snt.BatchNorm()(outputs, is_training=True)
            outputs = tf.nn.relu(outputs)

            outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(outputs)
            outputs = snt.BatchNorm()(outputs, is_training=True)
            outputs = tf.nn.relu(outputs)


            outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(outputs)
            outputs = snt.BatchNorm()(outputs, is_training=True)
            outputs = tf.nn.relu(outputs)



            outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(outputs)
            outputs = snt.BatchNorm()(outputs, is_training=True) # todo: deal with train/test time
            outputs = tf.nn.relu(outputs)


            outputs = snt.BatchFlatten()(outputs)
            #outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time
            outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(outputs)

            return outputs

        return convnet1d

    @staticmethod
    def make_transpose_cnn_model():
        def transpose_convnet1d(inputs):
            inputs = tf.expand_dims(inputs, axis=2)

            outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(inputs)
            outputs = snt.BatchNorm()(outputs, is_training=True)
            outputs = tf.nn.relu(outputs)

            outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(outputs)
            outputs = snt.BatchNorm()(outputs, is_training=True)
            outputs = tf.nn.relu(outputs)

            outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(outputs)
            outputs = snt.BatchNorm()(outputs, is_training=True)
            outputs = tf.nn.relu(outputs)

            outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                          kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=1)(outputs)
            outputs = snt.BatchNorm()(outputs, is_training=True) #todo: deal with train/test time
            outputs = tf.nn.relu(outputs)

            outputs = snt.BatchFlatten()(outputs)
            #outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time
            outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(outputs)

            return outputs

        return transpose_convnet1d
