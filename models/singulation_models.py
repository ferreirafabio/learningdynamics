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
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Flatten, Dense, BatchNormalization, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.optimizers import SGD
from utils.utils import get_correct_image_shape
from tensorflow.contrib.slim.nets import resnet_v1


import keras
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
                global_model_fn=EncodeProcessDecode.make_mlp_model
            )

    def _build(self, inputs):
        return self._network(inputs)


class CNNEncoderGraphIndependent(snt.AbstractModule):
    """GraphNetwork with CNN node and MLP edge / global models."""

    def __init__(self, name="CNNEncoderGraphIndependent"):
        super(CNNEncoderGraphIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
              edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
              node_model_fn=lambda: ResNet50Encoder(name='cnn_nodes'),#lambda inputs, is_training: Layer5ConvNet1D(name='cnn_nodes')(inputs, is_training),
              global_model_fn=lambda: ResNet50Encoder(name='cnn_globals')#lambda inputs, is_training: Layer5ConvNet1D(name='cnn_globals')(inputs, is_training)
            )

    def _build(self, inputs):
        return self._network(inputs)


class CNNDecoderGraphIndependent(snt.AbstractModule):
    """Graph decoder network with Transpose CNN node and MLP edge / global models."""

    def __init__(self, name="CNNDecoderGraphIndependent"):
        super(CNNDecoderGraphIndependent, self).__init__(name=name)

        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=EncodeProcessDecode.make_mlp_model_edges,
                node_model_fn=lambda: TransposeLayer5ConvNet2D(name="transpose_nodes"),#lambda inputs, is_training: TransposeLayer5ConvNet1D(name="2dconvdecoder_nodes")(inputs=inputs, is_training=is_training),
                global_model_fn=lambda: TransposeLayer5ConvNet2D(name="transpose_globals")#lambda inputs, is_training: TransposeLayer5ConvNet1D(name="2dconvdecoder_globals")(inputs=inputs, is_training=is_training)
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
        EncodeProcessDecode.n_neurons = config.n_neurons
        EncodeProcessDecode.n_neurons_edges = config.n_neurons_edges
        EncodeProcessDecode.n_layers_edges = config.n_layers_edges
        EncodeProcessDecode.n_convnet1D_filters_per_layer = config.n_convnet1D_filters_per_layer
        EncodeProcessDecode.convnet1D_kernel_size = config.convnet1D_kernel_size
        EncodeProcessDecode.convnet1D_stride = config.convnet1D_stride
        EncodeProcessDecode.convnet1D_pooling = config.convnet1D_pooling
        EncodeProcessDecode.convnet1D_tanh = config.convnet1D_tanh
        EncodeProcessDecode.depth_data_provided = config.depth_data_provided
        EncodeProcessDecode.n_neurons_mlp_position_velocity = config.n_neurons_mlp_position_velocity

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

        self.init_transform()

    def _build(self, input_op, num_processing_steps, is_training):
        latent = self._encoder(input_op)#, is_training)

        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(input_op)#, is_training)
            output_ops.append(self._output_transform(decoded_op))

        return output_ops

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
                tf.losses.mean_squared_error(output_op.nodes[:, -6:], node_splits[i][:, -6:]) +
                tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
                    for i, output_op in enumerate(output_ops)
            ]
        else:
            loss_ops = [
                tf.losses.mean_squared_error(output_op.nodes, node_splits[i]) +
                tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
                for i, output_op in enumerate(output_ops)
            ]

        pos_vel_loss_ops = [
            tf.losses.mean_squared_error(output_op.nodes[:, -6:], node_splits[i][:, -6:])
            for i, output_op in enumerate(output_ops)
        ]

        # todo: might use weighted MSE loss here
        # todo: perhaps include global attributes into loss function

        return loss_ops, pos_vel_loss_ops


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
        return snt.Sequential([snt.nets.MLP([EncodeProcessDecode.n_neurons] * EncodeProcessDecode.n_layers, activate_final=True),
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



def make_pos_vel_encoder_model(input):
    return snt.Sequential([snt.nets.MLP([6, EncodeProcessDecode.n_neurons_mlp_position_velocity], activate_final=True), snt.LayerNorm()])(input)


def make_pos_vel_decoder_model(input):
    return snt.Sequential(
        [snt.nets.MLP([EncodeProcessDecode.n_neurons_mlp_position_velocity, 6], activate_final=True), snt.LayerNorm()])(input)


class Layer5ConvNet1D(snt.AbstractModule):
    def __init__(self, name='cnn_model'):
        super(Layer5ConvNet1D, self).__init__(name=name)
        # todo:  store parameters in member variables prefixed with an underscore, to indicate that it is private.

    def _build(self, inputs, is_training):
        if EncodeProcessDecode.convnet1D_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu
        # input shape is (batch_size, feature_length) but CNN operates on depth channels --> (batch_size, feature_length, 1)
        inputs = tf.expand_dims(inputs, axis=2)
        ''' layer 1'''
        outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                             kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(inputs)

        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        if EncodeProcessDecode.convnet1D_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)

        ''' layer 2'''
        outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                             kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        if EncodeProcessDecode.convnet1D_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)

        ''' layer 3'''
        outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                             kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        if EncodeProcessDecode.convnet1D_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)

        ''' layer 4'''
        outputs = snt.Conv1D(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                             kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)  # todo: deal with train/test time
        if EncodeProcessDecode.convnet1D_pooling:
            outputs = tf.layers.max_pooling1d(outputs, 2, 2)
        outputs = activation(outputs)


        ''' layer 5'''
        outputs = snt.BatchFlatten()(outputs)
        #outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time
        outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(outputs)

        return outputs


class TransposeLayer5ConvNet1D(snt.AbstractModule):
    def __init__(self, name='transpose_cnn_model'):
        super(TransposeLayer5ConvNet1D, self).__init__(name=name)

    def _build(self, inputs, is_training):
        if EncodeProcessDecode.convnet1D_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu

        inputs = tf.expand_dims(inputs, axis=2)

        ''' layer 1'''
        outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                      kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(
            inputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)

        ''' layer 2'''
        outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                      kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(
            outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)

        ''' layer 3'''
        outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                      kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(
            outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)
        outputs = activation(outputs)

        ''' layer 4'''
        outputs = snt.Conv1DTranspose(output_channels=EncodeProcessDecode.n_convnet1D_filters_per_layer,
                                      kernel_shape=EncodeProcessDecode.convnet1D_kernel_size, stride=EncodeProcessDecode.convnet1D_stride)(
            outputs)
        outputs = snt.BatchNorm()(outputs, is_training=is_training)  # todo: deal with train/test time
        outputs = activation(outputs)

        ''' layer 5'''
        outputs = snt.BatchFlatten()(outputs)
        # outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time
        outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(outputs)

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
            vel_pos_output = snt.Sequential([snt.nets.MLP([6, EncodeProcessDecode.n_neurons_mlp_position_velocity], activate_final=True), snt.LayerNorm()])(vel_pos)
            #vel_pos_output = make_pos_vel_encoder_model(vel_pos)

            outputs = keras.layers.concatenate([features_rgb, features_seg, features_depth, vel_pos_output], axis=1)


            # todo: save ResNet weights
            # https://github.com/sebastianbk/finetuned-resnet50-keras/blob/master/resnet50_train.py

            #outputs = snt.BatchFlatten()(outputs)
            #print("Encoder Output Shape", outputs.get_shape())
            outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(outputs)
            #outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(features_rgb)
            return outputs


class TransposeLayer5ConvNet2D(snt.AbstractModule):
    def __init__(self, name='transpose_convnet2d'):
        super(TransposeLayer5ConvNet2D, self).__init__(name=name)

    def _build(self, inputs, is_training=True):

        if EncodeProcessDecode.convnet1D_tanh:
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu

        img_shape = get_correct_image_shape(config=None, get_type='all', depth_data_provided=EncodeProcessDecode.depth_data_provided)

        #print("Decoder Input Shape", inputs.get_shape())
        """ separate image from position/velocity data """
        pos_vel_latent = inputs[:, -EncodeProcessDecode.n_neurons_mlp_position_velocity:] # get last n elements
        image_data = inputs[:, :-EncodeProcessDecode.n_neurons_mlp_position_velocity] # get everything except last n elements

        """ map latent position+velocity from 32d to original 6d space """
        #pos_vel_output = make_pos_vel_encoder_model(pos_vel_latent)
        pos_vel_output = snt.Sequential([snt.nets.MLP([EncodeProcessDecode.n_neurons_mlp_position_velocity, 6], activate_final=True), snt.LayerNorm()])(pos_vel_latent)

        """ in order to apply 1x1 2D convolutions, transform shape (batch_size, features) -> shape (batch_size, 1, 1, features)"""
        image_data = tf.expand_dims(image_data, axis=1)
        image_data = tf.expand_dims(image_data, axis=1)  # yields shape (?,1,1,128)

        print(image_data.get_shape())
        ''' layer 1 (1,1,x) -> (7,7,x) '''
        outputs = tf.layers.conv2d_transpose(image_data, filters=image_data.get_shape()[3], kernel_size=[1, 1], strides=(5, 5), padding='valid')
        #outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        print(outputs.get_shape())

        ''' layer 2 (7,7,x) -> (15,20,x) '''
        outputs = tf.layers.conv2d_transpose(outputs, filters=64, kernel_size=(3, 4), strides=(3, 4), padding='valid') #kernel_size=[3, 2], strides=(2, 3), padding='valid')
        #outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        print(outputs.get_shape())

        outputs = tf.layers.conv2d_transpose(outputs, filters=32, kernel_size=[2, 2], strides=(2, 2), padding='valid')
        #outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        print(outputs.get_shape())


        outputs = tf.layers.conv2d_transpose(outputs, filters=16, kernel_size=[2, 2], strides=(2, 2), padding='valid')
        #outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        print(outputs.get_shape())


        outputs = tf.layers.conv2d_transpose(outputs, filters=img_shape[2], kernel_size=[2, 2], strides=(2, 2), padding='valid')
        #outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = activation(outputs)
        print(outputs.get_shape())

        # ''' layer 1'''
        # outputs = snt.Conv2DTranspose(output_channels=image_data.get_shape()[3], kernel_shape=(1,1), stride=(5,5), padding='VALID')(image_data)
        # outputs = snt.BatchNorm()(outputs, is_training=is_training)
        # outputs = activation(outputs)
        # print(outputs.get_shape())
        #
        #
        # ''' layer 2'''
        # outputs = snt.Conv2DTranspose(output_channels=64, kernel_shape=(3,4), stride=(3,4), padding='VALID')(outputs)
        # outputs = snt.BatchNorm()(outputs, is_training=is_training)
        # outputs = activation(outputs)
        # print(outputs.get_shape())
        #
        # ''' layer 3'''
        # outputs = snt.Conv2DTranspose(output_channels=32, kernel_shape=(2,2), stride=(2,2), padding='VALID')(outputs)
        # outputs = snt.BatchNorm()(outputs, is_training=is_training)
        # outputs = activation(outputs)
        # print(outputs.get_shape())
        #
        # ''' layer 4'''
        # outputs = snt.Conv2DTranspose(output_channels=16, kernel_shape=(2,2), stride=(2,2), padding='VALID')(outputs)
        # outputs = snt.BatchNorm()(outputs, is_training=is_training)
        # outputs = activation(outputs)
        # print(outputs.get_shape())
        #
        # ''' layer 5'''
        # outputs = snt.Conv2DTranspose(output_channels=img_shape[2], kernel_shape=(2, 2), stride=(2, 2), padding='VALID')(outputs)
        # outputs = snt.BatchNorm()(outputs, is_training=is_training)
        # outputs = activation(outputs)
        # print(outputs.get_shape())

        ''' layer 6'''
        outputs = snt.BatchFlatten()(outputs)
        # outputs = tf.nn.dropout(outputs, keep_prob=tf.constant(1.0)) # todo: deal with train/test time

        outputs = tf.concat([outputs, pos_vel_output], axis=1)
        #print("Decoder Output Shape", outputs.get_shape())
        outputs = snt.Linear(output_size=EncodeProcessDecode.n_neurons)(outputs)
        return outputs
