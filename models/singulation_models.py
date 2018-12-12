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
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from utils.utils import get_correct_image_shape, is_square
from tensorflow.contrib.slim.nets import resnet_v1

import keras
import sonnet as snt
import tensorflow as tf
import math




class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
                node_model_fn=EncodeProcessDecode.make_mlp_model,
                global_model_fn=EncodeProcessDecode.make_mlp_model
            )

    def _build(self, inputs):
        return self._network(inputs)


class CNNEncoderGraphIndependent(snt.AbstractModule):
    """GraphNetwork with CNN node and MLP edge / global models."""

    def __init__(self, model_id, name="CNNEncoderGraphIndependent"):
        super(CNNEncoderGraphIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            """ we want to re-use the cnn encoder for both nodes and global attributes """
            visual_encoder = get_model_from_config(model_id, model_type="visual_encoder")(name="visual_encoder")
            self._network = modules.GraphIndependent(
              edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
              node_model_fn=lambda: get_model_from_config(model_id, model_type="non_visual_encoder")(visual_encoder, name="non_visual_enc_node"),
              global_model_fn=lambda: get_model_from_config(model_id, model_type="non_visual_encoder")(visual_encoder, name="non_visual_enc_global")
            )

    def _build(self, inputs):
        return self._network(inputs)


class CNNDecoderGraphIndependent(snt.AbstractModule):
    """Graph decoder network with Transpose CNN node and MLP edge / global models."""

    def __init__(self, model_id, name="CNNDecoderGraphIndependent"):
        super(CNNDecoderGraphIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            visual_decoder = get_model_from_config(model_id, model_type="visual_decoder")(name="visual_decoder")
            self._network = modules.GraphIndependent(
                edge_model_fn=EncodeProcessDecode.make_mlp_model_edges_decode,
                node_model_fn=lambda: get_model_from_config(model_id, model_type="non_visual_decoder")(visual_decoder, name="non_visual_dec_node"),
                global_model_fn=lambda: get_model_from_config(model_id, model_type="non_visual_decoder")(visual_decoder, name="non_visual_dec_global"),
            )

    def _build(self, inputs):
        return self._network(inputs)


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
        EncodeProcessDecode.dimensions_latent_repr = config.dimensions_latent_repr
        EncodeProcessDecode.n_neurons_edges = config.n_neurons_edges
        EncodeProcessDecode.n_layers_edges = config.n_layers_edges
        EncodeProcessDecode.convnet_pooling = config.convnet_pooling
        EncodeProcessDecode.convnet_tanh = config.convnet_tanh
        EncodeProcessDecode.depth_data_provided = config.depth_data_provided
        EncodeProcessDecode.n_neurons_mlp_nonvisual = config.n_neurons_mlp_nonvisual
        EncodeProcessDecode.n_conv_filters = config.n_conv_filters
        EncodeProcessDecode.edge_output_size = config.edge_output_size
        EncodeProcessDecode.model_type = config.model_type

        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        # init the batch counter
        self.init_batch_step()

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.use_cnn = self.config.node_and_global_as_cnn

        if self.use_cnn:
            self._encoder = CNNEncoderGraphIndependent(config.model_type)
            self._decoder = CNNDecoderGraphIndependent(config.model_type)
        else:
            self._encoder = MLPGraphIndependent()
            self._decoder = MLPGraphIndependent()

        self._core = MLPGraphNetwork()

        self.init_ops()

        self.node_output_size = config.node_output_size
        self.edge_output_size = config.edge_output_size
        self.global_output_size = config.global_output_size

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        self.init_transform()

    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)#, is_training)

        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)#, is_training)
            #output_ops.append(self._output_transform(decoded_op))
            output_ops.append(decoded_op)

        return output_ops

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
        self.loss_op_train = None
        self.loss_op_test = None

        self.loss_ops_train = None
        self.loss_ops_test = None

        self.pos_vel_loss_ops_test = None
        self.pos_vel_loss_ops_train = None

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


    @staticmethod
    def make_mlp_model():
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
          A Sonnet module which contains the MLP and LayerNorm.
        """
        return snt.Sequential([snt.nets.MLP([EncodeProcessDecode.dimensions_latent_repr] * EncodeProcessDecode.n_layers, activate_final=True),
                               snt.LayerNorm()])

    @staticmethod
    # since edges are very low-dim, use different number of neurons and layers
    def make_mlp_model_edges():
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
          A Sonnet module which contains the MLP and LayerNorm.
        """
        return snt.Sequential([snt.nets.MLP([EncodeProcessDecode.n_neurons_edges] * EncodeProcessDecode.n_layers_edges, activate_final=True),
                               snt.LayerNorm()])

    @staticmethod
    def make_mlp_model_edges_decode():
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
          A Sonnet module which contains the MLP and LayerNorm.
        """
        net = snt.nets.MLP([EncodeProcessDecode.n_neurons_edges] * EncodeProcessDecode.n_layers_edges, activate_final=True)
        return snt.Sequential([net, snt.Linear(EncodeProcessDecode.edge_output_size, name="edge_output")])


class Encoder5LayerConvNet1D(snt.AbstractModule):
    def __init__(self, name='cnn_model'):
        super(Encoder5LayerConvNet1D, self).__init__(name=name)

    def _build(self, inputs, is_training=True):
        if EncodeProcessDecode.convnet_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu
        # input shape is (batch_size, feature_length) but CNN operates on depth channels --> (batch_size, feature_length, 1)
        inputs = tf.expand_dims(inputs, axis=2)
        ''' layer 1'''
        outputs = snt.Conv1D(output_channels=12,
                             kernel_shape=10, stride=2)(inputs)

        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 2'''
        outputs = snt.Conv1D(output_channels=12,
                             kernel_shape=10, stride=2)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 3'''
        outputs = snt.Conv1D(output_channels=12,
                             kernel_shape=10, stride=2)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 4'''
        outputs = snt.Conv1D(output_channels=12,
                             kernel_shape=10, stride=2)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)  # todo: deal with train/test time
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 5'''
        outputs = snt.BatchFlatten()(outputs)
        #outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time
        outputs = snt.Linear(output_size=EncodeProcessDecode.dimensions_latent_repr)(outputs)
        #print(outputs.get_shape())
        return outputs


class Decoder5LayerConvNet1D(snt.AbstractModule):
    def __init__(self, name='transpose_cnn_model'):
        super(Decoder5LayerConvNet1D, self).__init__(name=name)

    def _build(self, inputs, is_training=True):
        if EncodeProcessDecode.convnet_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu

        inputs = tf.expand_dims(inputs, axis=2)

        ''' layer 1'''
        outputs = snt.Conv1DTranspose(output_channels=12, kernel_shape=10, stride=2)(inputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 2'''
        outputs = snt.Conv1DTranspose(output_channels=12, kernel_shape=10, stride=2)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 3'''
        outputs = snt.Conv1DTranspose(output_channels=12,
                                      kernel_shape=10, stride=2)(
            outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 4'''
        outputs = snt.Conv1DTranspose(output_channels=12, kernel_shape=10, stride=2)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)
        #print(outputs.get_shape())

        ''' layer 5'''
        outputs = snt.BatchFlatten()(outputs)
        # outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time
        outputs = snt.Linear(output_size=EncodeProcessDecode.dimensions_latent_repr)(outputs)
        #print(outputs.get_shape())

        return outputs


class Decoder5LayerConvNet2D(snt.AbstractModule):
    def __init__(self, name='decoder_convnet2d'):
        super(Decoder5LayerConvNet2D, self).__init__(name=name)

    def _build(self, inputs, is_training=True, verbose=False):
        filter_sizes = [EncodeProcessDecode.n_conv_filters, EncodeProcessDecode.n_conv_filters * 2]

        if EncodeProcessDecode.convnet_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu

        img_shape = get_correct_image_shape(config=None, get_type='all', depth_data_provided=EncodeProcessDecode.depth_data_provided)

        """ get image data, get everything >except< last n elements which are non-visual """
        image_data = inputs[:, :-EncodeProcessDecode.n_neurons_mlp_nonvisual]

        visual_latent_space_dim = EncodeProcessDecode.dimensions_latent_repr - EncodeProcessDecode.n_neurons_mlp_nonvisual

        """ in order to apply 1x1 2D convolutions, transform shape (batch_size, features) -> shape (batch_size, 1, 1, features)"""
        image_data = tf.expand_dims(image_data, axis=1)
        image_data = tf.expand_dims(image_data, axis=1)  # yields shape (?,1,1,latent_dim)

        #assert is_square(visual_latent_space_dim), "dimension of visual latent space vector (dimensions of latent representation: ({}) - " \
        #                                           "dimensions of non visual latent representation({})) must be square".format(
        #    EncodeProcessDecode.dimensions_latent_repr, EncodeProcessDecode.n_neurons_mlp_nonvisual)

        #image_data = tf.reshape(image_data, (-1, int(math.sqrt(visual_latent_space_dim)), int(math.sqrt(visual_latent_space_dim)), 1))
        image_data = tf.reshape(image_data, (-1, 7, 10, 5))

        ''' layer 1 (7,10,5) -> (7,10,filter_sizes[1])'''
        outputs = tf.layers.conv2d_transpose(image_data, filters=filter_sizes[1], kernel_size=3, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l1_shape = outputs.get_shape()

        ''' layer 2 (7,10,filter_sizes[1]) -> (15,20,filter_sizes[1]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[1], kernel_size=(3, 2), strides=2, padding='valid')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l2_shape = outputs.get_shape()

        ''' layer 2 (15,20,filter_sizes[1]) -> (15,20,filter_sizes[1]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[1], kernel_size=3, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l3_shape = outputs.get_shape()

        ''' layer 2 (15,20,filter_sizes[1]) -> (30,40,filter_sizes[1]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[1], kernel_size=3, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l4_shape = outputs.get_shape()

        ''' layer 3 (30,40,filter_sizes[1]) -> (30,40,filter_sizes[1]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[1], kernel_size=3, strides=2, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l5_shape = outputs.get_shape()


        ''' layer 4 (30,40,filter_sizes[1]) -> (30,40,filter_sizes[0]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[0], kernel_size=3, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l6_shape = outputs.get_shape()


        ''' layer 5 (30,40,filter_sizes[0]) -> (60,80,filter_sizes[0]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[0], kernel_size=3, strides=2, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l7_shape = outputs.get_shape()

        ''' layer 5 (60,80,filter_sizes[0]) -> (60,80,filter_sizes[0]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[0], kernel_size=3, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l8_shape = outputs.get_shape()

        ''' layer 5 (60,80,filter_sizes[0]) -> (120,160,filter_sizes[0]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[0], kernel_size=3, strides=2, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l9_shape = outputs.get_shape()

        ''' layer 5 (120,160,filter_sizes[0]) -> (120,160,filter_sizes[0]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=filter_sizes[0], kernel_size=3, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l10_shape = outputs.get_shape()

        ''' layer 6 (120,160,filter_sizes[0]) -> (120,160,3 or 4 or 7]) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=img_shape[2], kernel_size=1, strides=1, padding='same')
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        l11_shape = outputs.get_shape()

        visual_latent_output = tf.layers.flatten(outputs)

        if verbose:
            print("Image data shape", image_data.get_shape())
            print("Layer1 decoder output shape", l1_shape)
            print("Layer2 decoder output shape", l2_shape)
            print("Layer3 decoder output shape", l3_shape)
            print("Layer4 decoder output shape", l4_shape)
            print("Layer5 decoder output shape", l5_shape)
            print("Layer6 decoder output shape", l6_shape)
            print("Layer7 decoder output shape", l7_shape)
            print("Layer8 decoder output shape", l8_shape)
            print("Layer9 decoder output shape", l9_shape)
            print("Layer10 decoder output shape", l10_shape)
            print("Layer11 decoder output shape", l11_shape)
            print("decoder shape before adding non-visual data", visual_latent_output.get_shape())

        # outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time

        return visual_latent_output


class Encoder5LayerConvNet2D(snt.AbstractModule):
    def __init__(self, name):
        super(Encoder5LayerConvNet2D, self).__init__(name=name)

    def _build(self, inputs, name, is_training=True, verbose=False):

        if EncodeProcessDecode.convnet_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu

        if "global" in name:
            n_non_visual_elements = 5
        else:
            n_non_visual_elements = 6


        filter_sizes = [EncodeProcessDecode.n_conv_filters, EncodeProcessDecode.n_conv_filters * 2]

        img_data = inputs[:, :-n_non_visual_elements]  # shape: (batch_size, features)
        img_shape = get_correct_image_shape(config=None, get_type="all", depth_data_provided=EncodeProcessDecode.depth_data_provided)
        img_data = tf.reshape(img_data, [-1, *img_shape])  # -1 means "all", i.e. batch dimension

        ''' layer 1'''
        outputs = tf.layers.conv2d(img_data, filters=filter_sizes[0], kernel_size=3, strides=1, padding='same', activation=activation)
        outputs = activation(outputs)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        l1_shape = outputs.get_shape()

        ''' layer 2'''
        outputs = tf.layers.conv2d(outputs, filters=filter_sizes[0], kernel_size=3, strides=1, padding='same', activation=activation)
        outputs = activation(outputs)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        l2_shape = outputs.get_shape()

        ''' layer 3'''
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling2d(outputs, 2, 2)
        l3_shape = outputs.get_shape()

        ''' layer 4'''
        outputs = tf.layers.conv2d(outputs, filters=filter_sizes[0], kernel_size=3, strides=1, padding='same', activation=activation)
        outputs = activation(outputs)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        l4_shape = outputs.get_shape()

        ''' layer 5'''
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling2d(outputs, 2, 2)
        l5_shape = outputs.get_shape()


        ''' layer 6'''
        outputs = tf.layers.conv2d(outputs, filters=filter_sizes[1], kernel_size=3, strides=1, padding='same', activation=activation)
        outputs = activation(outputs)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        l6_shape = outputs.get_shape()

        ''' layer 7'''
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling2d(outputs, 2, 2)
        l7_shape = outputs.get_shape()

        ''' layer 8'''
        outputs = tf.layers.conv2d(outputs, filters=filter_sizes[1], kernel_size=3, strides=1, padding='same', activation=activation)
        outputs = activation(outputs)
        l8_shape = outputs.get_shape()

        ''' layer 9'''
        outputs = tf.layers.conv2d(outputs, filters=filter_sizes[1], kernel_size=3, strides=1, padding='same', activation=activation)
        outputs = activation(outputs)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        l9_shape = outputs.get_shape()

        ''' layer 10'''
        if EncodeProcessDecode.convnet_pooling:
            outputs = tf.layers.max_pooling2d(outputs, 2, 2)
        l10_shape = outputs.get_shape()

        outputs = tf.layers.batch_normalization(outputs, training=is_training)

        if verbose:
            print("Layer0 encoder output shape", l1_shape)
            print("Layer1 encoder output shape", l2_shape)
            print("Layer2 encoder output shape", l3_shape)
            print("Layer3 encoder output shape", l4_shape)
            print("Layer4 encoder output shape", l5_shape)
            print("Layer5 encoder output shape", l6_shape)
            print("Layer6 encoder output shape", l7_shape)
            print("Layer7 encoder output shape", l8_shape)
            print("Layer8 encoder output shape", l9_shape)
            print("Layer9 encoder output shape", l10_shape)

        ' shape (?, 7, 10, 32) -> (?, dimensions_latent_repr-n_neurons_mlp_nonvisual) '
        visual_latent_output = tf.layers.flatten(outputs)

        ''' layer 11'''
        visual_latent_output = tf.layers.dense(inputs=visual_latent_output, units=EncodeProcessDecode.dimensions_latent_repr - EncodeProcessDecode.n_neurons_mlp_nonvisual)
        return visual_latent_output


class NonVisualDecoder(snt.AbstractModule):
    def __init__(self, visual_dec, name='NonVisualEncoder'):
        super(NonVisualDecoder, self).__init__(name=name)
        self.visual_dec = visual_dec
        self._name = name

    def _build(self, inputs, is_training=True):

        visual_decoded_output = self.visual_dec(inputs)
        print(visual_decoded_output.get_shape())

        if "global" in self._name:
            n_non_visual_elements = 5
            non_visual_latent_output = inputs[:, -n_non_visual_elements:]  # get x,y,z-position, time-step and gravity constant

        else:
            n_non_visual_elements = 6
            non_visual_latent_output = inputs[:, -n_non_visual_elements:]  # get x,y,z-position and x,y,z-velocity

        """ map latent position/velocity (nodes) or position/gravity/time-step (global) from 32d to original 5d/6d space """
        non_visual_decoded_output = snt.Sequential([snt.nets.MLP([EncodeProcessDecode.n_neurons_mlp_nonvisual, n_non_visual_elements], activate_final=True), snt.LayerNorm()])(non_visual_latent_output)

        outputs = tf.concat([visual_decoded_output, non_visual_decoded_output], axis=1)
        #print("final decoder output shape after including non-visual data", outputs.get_shape())

        return outputs


class NonVisualEncoder(snt.AbstractModule):
    def __init__(self, visual_enc, name='NonVisualEncoder'):
        super(NonVisualEncoder, self).__init__(name=name)
        self._visual_enc = visual_enc
        self._name = name

    def _build(self, inputs, is_training=True):

        visual_latent_output = self._visual_enc(inputs, name=self._name)

        if "global" in self._name:
            n_non_visual_elements = 5
            non_visual_elements = inputs[:, -n_non_visual_elements:]  # get x,y,z-position, time-step and gravity constant
        else:
            n_non_visual_elements = 6
            non_visual_elements = inputs[:, -n_non_visual_elements:]  # get x,y,z-position and x,y,z-velocity

        """ map velocity and position into a latent space, concatenate with visual latent space vector """
        non_visual_latent_output = snt.Sequential([snt.nets.MLP([n_non_visual_elements, EncodeProcessDecode.n_neurons_mlp_nonvisual], activate_final=True), snt.LayerNorm()])(non_visual_elements)
        outputs = tf.concat([visual_latent_output, non_visual_latent_output], axis=1)
        #print("final decoder output shape", outputs.get_shape())
        return outputs


class ResNet50Encoder(snt.AbstractModule):
    def __init__(self, name='resnet50encoder'):
        super(ResNet50Encoder, self).__init__(name=name)

    def _build(self, inputs, is_training=True):
        """
        Extracts ResNet50 features from the given images and concatenates them into one large latent vector. If 3 image types are used
        (RGB, Segmentation and Depth), a vector of dimensionality (batch_size, config.n_neurons * 3 large + 6 (position+velocity)) is
        returned. The pre-trained ResNet50 on ImageNet is used and as features, a new fully connected layer is learned.
        :param inputs:
        :param is_training:
        :return:
        """
        img_data = inputs[:, :-6] # shape: (batch_size, features)
        img_shape = get_correct_image_shape(config=None, get_type="all", depth_data_provided=EncodeProcessDecode.depth_data_provided)

        img_data = tf.reshape(img_data, [-1, *img_shape]) #-1 means "all", i.e. batch dimension
        input_rgb = img_data[..., :3]
        input_seg = img_data[..., 4]

        with self._enter_variable_scope():
            #with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            base_model = VGG16(weights='imagenet', pooling='max', include_top=False, input_shape=(120, 160, 3))

            """ prepare ResNet-50 model for fine-tuning: freeze all layers, add one to-be-learned fc-layer """
            base_model.layers.pop()
            for layer in base_model.layers:
               layer.trainable = False
            last = base_model.layers[-1].output  # pool 5 output
            last = Flatten()(last)
            output = Dense(512, activation='relu')(last)

            finetuned_model = Model(inputs=base_model.input, outputs=output)
            #finetuned_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mse') # todo: check if this line is necessary

            x = preprocess_input(input_rgb)  # ResNet requires data to be centered
            features_rgb = finetuned_model(x)

            input_seg = tf.stack([input_seg]*3, axis=-1)
            x = preprocess_input(input_seg)
            features_seg = finetuned_model(x)

            if EncodeProcessDecode.depth_data_provided:
                input_depth = img_data[..., -3:]
                x = preprocess_input(input_depth)
                features_depth = finetuned_model(x)

            """ map velocity and map into a latent space """
            vel_pos = inputs[:, -6:]
            vel_pos_output = snt.Sequential([snt.nets.MLP([6, EncodeProcessDecode.n_neurons_mlp_nonvisual], activate_final=True), snt.LayerNorm()])(vel_pos)
            #vel_pos_output = make_pos_vel_encoder_model(vel_pos)

            outputs = keras.layers.concatenate([features_rgb, features_seg, features_depth, vel_pos_output], axis=1)

            # todo: save ResNet weights
            # https://github.com/sebastianbk/finetuned-resnet50-keras/blob/master/resnet50_train.py

            #outputs = snt.BatchFlatten()(outputs)
            #print("Encoder Output Shape", outputs.get_shape())
            outputs = snt.Linear(output_size=EncodeProcessDecode.dimensions_latent_repr)(outputs)
            #outputs = snt.Linear(output_size=EncodeProcessDecode.dimensions_latent_repr)(features_rgb)
            return outputs


def get_model_from_config(model_id, model_type="encoder"):
    """ cnn2d case """
    if "cnn2d" in model_id and model_type == "visual_encoder":
        return Encoder5LayerConvNet2D
    if "cnn2d" in model_id and model_type == "non_visual_encoder":
        return NonVisualEncoder
    if "cnn2d" in model_id and model_type == "visual_decoder":
        return Decoder5LayerConvNet2D
    if "cnn2d" in model_id and model_type == "non_visual_decoder":
        return NonVisualDecoder

    """ cnn1d case """
    if "cnn1d" in model_id and model_type == "encoder":
        return Encoder5LayerConvNet1D()
    if "cnn1d" in model_id and model_type == "decoder":
        return Decoder5LayerConvNet1D()


