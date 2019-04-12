import numpy as np
import tensorflow as tf
import tflearn
import sys
import os

#from data_loader.data_generator import DataGenerator
#from utils.config import process_config
#from utils.dirs import create_dirs
#from utils.utils import get_args
#from utils.conversions import convert_dict_to_list_subdicts

import tensorflow.contrib.slim.nets
#from tensorflow.contrib.slim.nets.resnet_v2 import resnet_v2_50


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

model_dirs = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../models')


def cnnmodel(inp_rgb, inp_stepsize=None, is_training=True):
    step = inp_stepsize
    step = tflearn.layers.core.fully_connected(step, 16, activation='relu')
    step = tflearn.layers.normalization.batch_normalization(step)
    step = tflearn.layers.core.fully_connected(step, 24, activation='relu')
    step = tflearn.layers.normalization.batch_normalization(step)
    step_low = tflearn.layers.core.fully_connected(step, 32, activation='relu')
    step = step_low
    step = tflearn.layers.normalization.batch_normalization(step)
    step = tflearn.layers.core.fully_connected(step, 49, activation='relu')   # changed 64 to 49
    step_high = tflearn.layers.normalization.batch_normalization(step)

    step_low = tf.reshape(step_low, (-1, 1, 1, 32))
    step_low = tf.tile(step_low, [1, 32, 32, 1])

    step_high = tf.reshape(step_high, (-1, 1, 1, 49))
    step_high = tf.tile(step_high, [1, 7, 7, 1])

    """ small encoder-cnn: """
    feat_e_0 = tflearn.layers.conv.conv_2d(inp_rgb, 10, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_0 = tflearn.layers.conv.max_pool_2d(feat_e_0, [2, 2], [2, 2], padding='VALID')
    feat_e_0 = tf.contrib.layers.layer_norm(feat_e_0)
    print(feat_e_0)

    feat_e_1 = tflearn.layers.conv.conv_2d(feat_e_0, 10, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_1 = tflearn.layers.conv.max_pool_2d(feat_e_1, [2, 2], [2, 2], padding='VALID')
    feat_e_1 = tf.contrib.layers.layer_norm(feat_e_1)
    print(feat_e_1)

    feat_e_2 = tflearn.layers.conv.conv_2d(feat_e_1, 5, (1, 1), strides=(1, 1), activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_2 = tflearn.layers.conv.max_pool_2d(feat_e_2, [2, 2], [2, 3], padding='SAME')
    feat_e_2 = tf.contrib.layers.layer_norm(feat_e_2)
    print(feat_e_2)

    feat_e_2 = tflearn.layers.conv.conv_2d(feat_e_2, 7, (1, 1), strides=(1, 1), activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_3 = tflearn.layers.conv.max_pool_2d(feat_e_2, [2, 2], [2, 2], padding='VALID')
    feat_e_3 = tf.contrib.layers.layer_norm(feat_e_3)
    print(feat_e_3)  # --> (?,7,7,7)

    #network_fn = get_network_fn('resnet_v2_50', num_classes=None, weight_decay=1e-5, is_training=is_training)
    #net, end_points = network_fn(inp_rgb)

    #net, end_points = tf.contrib.slim.nets.resnet_v2.resnet_v2_50(inp_rgb, num_classes=None, is_training=is_training)

    #feat_C4 = end_points['resnet_v2_50/block4']
    #feat_C3 = end_points['resnet_v2_50/block3']
    #feat_C2 = end_points['resnet_v2_50/block2']
    #feat_C1 = end_points['resnet_v2_50/block1']
    #print("feat_C4", feat_C4)
    #print("feat_C3", feat_C3)
    #print("feat_C2", feat_C2)
    #print("feat_C1", feat_C1)
    print("step_high", step_high)
    print("step_low", step_low)

    feat_C4 = tf.concat([feat_e_3, step_high], axis=-1)  # feat_C4 = tf.concat([feat_C4, step_high], axis=-1)
    print("feat_C4", feat_C4)

    feat = feat_C4
    feat = tflearn.layers.conv.conv_2d(feat, 7, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                       regularizer='L2')
    feat_0 = tflearn.layers.conv.avg_pool_2d(feat, [8, 8], [8, 8], padding='VALID')
    print(feat_0)

    feat_0 = tflearn.layers.conv.conv_2d(feat_0, 7, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                         regularizer='L2')
    feat_0 = tf.image.resize_bilinear(feat_0, [8, 8], align_corners=True)
    print(feat_0)

    feat_1 = tflearn.layers.conv.conv_2d(feat, 5, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                         regularizer='L2')
    feat_2 = tflearn.layers.conv.atrous_conv_2d(feat, 5, (3, 3), rate=2, activation='relu', weight_decay=1e-5,
                                                regularizer='L2')

    feat_4 = tflearn.layers.conv.atrous_conv_2d(feat, 10, (3, 3), rate=4, activation='relu', weight_decay=1e-5,
                                                regularizer='L2')

    feat_6 = tflearn.layers.conv.atrous_conv_2d(feat, 10, (3, 3), rate=6, activation='relu', weight_decay=1e-5,
                                                regularizer='L2')

    feat_ac = tf.concat([feat_0, feat_1, feat_4, feat_6], axis=-1)
    feat_ac = tflearn.layers.conv.conv_2d(feat_ac, 10, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                          regularizer='L2')
    feat_ac = tflearn.layers.core.dropout(feat_ac, keep_prob=0.9)

    feat_low = tf.concat([feat_C1, step_low], axis=-1)

    print("feat_low", feat_low)

    feat_high = tf.image.resize_bilinear(feat_ac, [32, 32], align_corners=True)

    feat_full = tf.concat([feat_high, feat_low], axis=-1)

    x = tflearn.layers.conv.conv_2d_transpose(feat_full, 128, [5, 5], [64, 64], strides=2, activation='relu',
                                              weight_decay=1e-5, regularizer='L2')
    x = tflearn.layers.normalization.batch_normalization(x)
    x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [128, 128], strides=2, activation='relu',
                                              weight_decay=1e-5, regularizer='L2')
    x = tflearn.layers.normalization.batch_normalization(x)
    x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [256, 256], strides=2, activation='relu',
                                              weight_decay=1e-5, regularizer='L2')
    x = tflearn.layers.normalization.batch_normalization(x)

    score = x
    score = tflearn.layers.conv.conv_2d(score, 2, (3, 3), strides=1, activation='linear', weight_decay=1e-3,
                                        regularizer='L2')
    return score


def main():
    # # create tensorflow session
    # sess = tf.Session()
    #
    # try:
    #     args = get_args()
    #     config = process_config(args.config)
    #
    #     config.old_tfrecords = args.old_tfrecords
    #     config.normalize_data = args.normalize_data
    #
    # except Exception as e:
    #     print("An error occurred during processing the configuration file")
    #     print(e)
    #     exit(0)
    #
    #
    # # create the experiments dirs
    # create_dirs([config.summary_dir, config.checkpoint_dir, config.config_file_dir])
    #
    # # create your data generator
    # train_data = DataGenerator(config, sess, train=True)
    # test_data = DataGenerator(config, sess, train=False)

    # prefix = config.exp_name
    # print("using {} rollout steps".format(config.n_rollouts))

    # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(init)

    # inp_rgb = tf.placeholder("float", [None, 120, 160, 3])
    # control = tf.placeholder("float", [None, 6])
    # gt_seg = tf.placeholder("float", [None, 120, 160])
    #
    # pred = cnnmodel(inp_rgb, control)

    #predictions = tf.reshape(pred, [-1, pred.get_shape()[1] * pred.get_shape()[2]])
    #labels = tf.reshape(gt_seg, [-1, gt_seg.get_shape()[1] * gt_seg.get_shape()[2]])

    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, predictions=predictions))
    #optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

    # cur_batch_it = 0
    # while True:
    #     try:
    #
    #         next_element = train_data.get_next_batch()
    #         features = sess.run(next_element)
    #
    #         features = convert_dict_to_list_subdicts(features, config.train_batch_size)
    #
    #
    #         with sess:
    #             for _ in range(config.train_batch_size):
    #                 rgb_batch = []
    #                 target_seg_batch = []
    #                 for feature in features:
    #                     for step in range(config.n_rollouts-1):
    #                         # get object 0
    #                         rgb_batch.append(feature["obect_segments"][step][3][:,:,:3])
    #                         target_seg = feature["obect_segments"][step][3][:,:,3] # get seg
    #
    #                         gripper_pos = feature["gripperpos"][step+1]
    #                         gripper_vel = feature["grippervel"][step+1]
    #                         gripper_pos_vel = np.concatenate([gripper_pos, gripper_vel])
    #
    #                         #loss = sess.run([optimizer, pred], feed_dict={inp_rgb: rgb_input, control: gripper_pos_vel, gt_seg: target_seg})
    #
    #
    #
    #         if cur_batch_it % config.test_interval == 1:
    #             print("Executing test batch")
    #             #test_batch(prefix, export_images=self.config.export_test_images,
    #             #                initial_pos_vel_known=self.config.initial_pos_vel_known,
    #             #                sub_dir_name="tests_during_training")
    #         cur_batch_it += 1
    #
    #     except tf.errors.OutOfRangeError:
    #         break
    #
    return None



if __name__ == '__main__':
    inp_rgb = tf.placeholder("float", [None, 120, 160, 3])
    control = tf.placeholder("float", [None, 6])
    gt_seg = tf.placeholder("float", [None, 120, 160])

    pred = cnnmodel(inp_rgb, control)

    main()


