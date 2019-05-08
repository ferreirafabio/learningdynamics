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

model_dirs = os.path.join(os.path.dirname(os.path.abspath(__file__)),'./.')

class baseline_auto_predictor(BaseModel):
    def __init__(self, config, name="baseline_auto_predictor"):
        super(baseline_auto_predictor, self).__init__(self)
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        # init the batch counter
        self.init_batch_step()

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

    def encoder(self, in_rgbsegxyz, is_training):
        x = in_rgbsegxyz
        """ Layer 1 """
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_1")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 2 """
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_2")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tflearn.layers.conv.max_pool_2d(x, 2, 2)

        """ Layer 3 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_3")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 4 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_4")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tflearn.layers.conv.max_pool_2d(x, 2, 2)

        """ Layer 5 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_5")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 6 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_6")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tflearn.layers.conv.max_pool_2d(x, 2, 2)

        """ Layer 7 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_7")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 8 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_8")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tflearn.layers.conv.max_pool_2d(x, 2, 2)

        """ Layer 9 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_9")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 10 """
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope="conv1_10")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tflearn.layers.conv.max_pool_2d(x, 2, 2)

        x = tflearn.layers.flatten(x)

        return x

    def decoder(self, latent, is_training):
        x = latent

        """ Layer 1 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[2, 2], nb_filter=256, filter_size=2, strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', padding='valid')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 2 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[4, 4], nb_filter=256, filter_size=2, strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 3 """
        x = tflearn.layers.conv.conv_2d_transpose(x, nb_filter=256, output_shape=[7, 10], filter_size=4, strides=[1, 2], activation='relu', weight_decay=1e-5, regularizer='L2', padding="valid")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 4 """
        x = tflearn.layers.conv.conv_2d_transpose(x, nb_filter=256, output_shape=[15, 20], filter_size=[3, 2], strides=2,  activation='relu', weight_decay=1e-5, regularizer='L2', padding="valid")
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 5 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[15, 20], nb_filter=256, filter_size=2, strides=1,  activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 6 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[15, 20], nb_filter=128, filter_size=2, strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 7 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[30, 40], nb_filter=128, filter_size=2, strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 8 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[30, 40], nb_filter=128, filter_size=3, strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 9 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[60, 80], nb_filter=128, filter_size=3, strides=2, activation='relu',  weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 10 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[60, 80], nb_filter=128, filter_size=3, strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 11 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[120, 160], nb_filter=64, filter_size=3, strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        """ Layer 12 """
        x = tflearn.layers.conv.conv_2d_transpose(x, output_shape=[120, 160], nb_filter=2, filter_size=3, strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.layers.batch_normalization(x, training=is_training)

        return x


    def cnnmodel(self, in_rgb, in_segxyz, in_control=None, is_training=True):
        ctrl = in_control
        ctrl = tflearn.layers.core.fully_connected(ctrl, 32, activation='relu')
        ctrl = tf.layers.batch_normalization(ctrl, training=is_training)

        ctrl = tflearn.layers.core.fully_connected(ctrl, 32, activation='relu')
        ctrl = tf.layers.batch_normalization(ctrl, training=is_training)
        ctrl = tflearn.layers.core.fully_connected(ctrl, 32, activation='relu')
        latent_ctrl = tf.layers.batch_normalization(ctrl, training=is_training)

        in_rgb_segxyz = tf.concat([in_rgb, in_segxyz], axis=-1)
        latent_img = self.encoder(in_rgbsegxyz=in_rgb_segxyz, is_training=is_training)

        latent_img_ctrl = tf.concat([latent_img, latent_ctrl], axis=-1)
        latent_img_ctrl = tflearn.layers.core.fully_connected(latent_img_ctrl, 256, activation='relu')

        latent_img_ctrl = tf.expand_dims(latent_img_ctrl, axis=1)
        latent_img_ctrl = tf.expand_dims(latent_img_ctrl, axis=1)

        score = self.decoder(latent=latent_img_ctrl, is_training=is_training)

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
