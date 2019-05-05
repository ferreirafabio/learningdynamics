import numpy as np
import tensorflow as tf
import tflearn
import sys
import os
from utils.utils import get_var_list_to_restore_by_name
from base.base_model import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR,'../')
sys.path.append(BASE_DIR)

from nets_factory import get_network_fn

model_dirs = os.path.join(os.path.dirname(os.path.abspath(__file__)),'./.')

class Model_naive_xyz(BaseModel):
    def __init__(self, config, name="Model_naive_xyz_net"):
        super(Model_naive_xyz, self).__init__(self)
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        # init the batch counter
        self.init_batch_step()

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

    def cnn(self, in_segxyz):
        x = in_segxyz
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_1")
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_2")
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_3")

        ##60,80
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv2_1")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv2_2")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv2_3")
        x = tflearn.layers.normalization.batch_normalization(x)

        ##30,40
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv3_1")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv3_2")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv3_3")
        x = tflearn.layers.normalization.batch_normalization(x)

        ##30,40
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv4_1")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv4_2")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv4_3")
        x = tflearn.layers.normalization.batch_normalization(x)

        ##10,20
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv5_1")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv5_2")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv5_3")
        return x

    def cnnmodel(self, in_rgb, in_segxyz, in_control=None, is_training=True):
        step = in_control
        step = tflearn.layers.core.fully_connected(step, 16, activation='relu')
        step = tflearn.layers.normalization.batch_normalization(step)
        step = tflearn.layers.core.fully_connected(step, 24, activation='relu')
        step = tflearn.layers.normalization.batch_normalization(step)
        step_low = tflearn.layers.core.fully_connected(step, 32, activation='relu')
        step = step_low
        step = tflearn.layers.normalization.batch_normalization(step)
        step = tflearn.layers.core.fully_connected(step, 64, activation='relu')
        step_high = tflearn.layers.normalization.batch_normalization(step)

        step_low = tf.reshape(step_low, (-1, 1, 1, 32))
        step_low = tf.tile(step_low, [1, 15, 20, 1])

        step_high = tf.reshape(step_high, (-1, 1, 1, 64))
        step_high = tf.tile(step_high, [1, 4, 5, 1])

        xyz_high = self.cnn(in_segxyz)

        network_fn = get_network_fn('resnet_v2_50', num_classes=None, weight_decay=1e-5, is_training=is_training)
        net, end_points = network_fn(in_rgb)

        feat_C4 = end_points['resnet_v2_50/block4']
        feat_C3 = end_points['resnet_v2_50/block3']
        feat_C2 = end_points['resnet_v2_50/block2']
        feat_C1 = end_points['resnet_v2_50/block1']
        print("feat_C4", feat_C4)
        print("feat_C1", feat_C1)

        feat_C4 = tf.concat([feat_C4, step_high, xyz_high], axis=-1)

        feat = feat_C4
        feat = tflearn.layers.conv.conv_2d(feat, 1024, (1, 1), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')

        print("feat", feat)
        feat_0 = tflearn.layers.conv.avg_pool_2d(feat, [4, 5], [4, 5], padding='VALID')
        feat_0 = tflearn.layers.conv.conv_2d(feat_0, 256, (1, 1), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        feat_0 = tf.image.resize_bilinear(feat_0, [4, 5], align_corners=True)

        feat_1 = tflearn.layers.conv.conv_2d(feat, 256, (1, 1), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')

        feat_2 = tflearn.layers.conv.atrous_conv_2d(feat, 256, (3, 3), rate=2, activation='relu', weight_decay=1e-5, regularizer='L2')

        feat_4 = tflearn.layers.conv.atrous_conv_2d(feat, 256, (3, 3), rate=4, activation='relu', weight_decay=1e-5, regularizer='L2')

        feat_6 = tflearn.layers.conv.atrous_conv_2d(feat, 256, (3, 3), rate=6, activation='relu', weight_decay=1e-5, regularizer='L2')

        feat_ac = tf.concat([feat_0, feat_1, feat_4, feat_6], axis=-1)
        feat_ac = tflearn.layers.conv.conv_2d(feat_ac, 256, (1, 1), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        feat_ac = tflearn.layers.core.dropout(feat_ac, keep_prob=0.9)

        feat_low = tf.concat([feat_C1, step_low], axis=-1)

        print("feat_low", feat_low)

        feat_high = tf.image.resize_bilinear(feat_ac, [15, 20], align_corners=True)

        feat_full = tf.concat([feat_high, feat_low], axis=-1)

        x = tflearn.layers.conv.conv_2d_transpose(feat_full, 128, [5, 5], [30, 40], strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.normalization.batch_normalization(x)
        x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [60, 80], strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.normalization.batch_normalization(x)
        x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [120, 160], strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.normalization.batch_normalization(x)

        score = x
        score = tflearn.layers.conv.conv_2d(score, 2, (3, 3), strides=1, activation='linear', weight_decay=1e-3, regularizer='L2')
        return score

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.cur_batch_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor+1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_batch_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.cur_batch_tensor = tf.Variable(0, trainable=False, name='cur_batch')
            self.increment_cur_batch_tensor = tf.assign(self.cur_batch_tensor, self.cur_batch_tensor+1)


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_checkpoints_to_keep)


    def load_resnet(self, sess):
        tf_initial_checkpoint_v2 = "../models/pretrained_nets/resnet_v2_50/resnet_v2_50.ckpt"
        variables_to_restore_v2 = get_var_list_to_restore_by_name('resnet_v2_50')
        restorer_v2 = tf.train.Saver(variables_to_restore_v2)
        restorer_v2.restore(sess, tf_initial_checkpoint_v2)
