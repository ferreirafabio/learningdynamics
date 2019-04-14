import numpy as np
import tensorflow as tf
import tflearn
import sys
import os


from data_loader.data_generator import DataGenerator
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.conversions import convert_dict_to_list_subdicts
from utils.io import save_to_gif_from_dict
from utils.utils import check_exp_folder_exists_and_create
from utils.math_ops import sigmoid

def cnnmodel(inp_rgb, control=None):
    step = control
    step = tflearn.layers.core.fully_connected(step, 16, activation='relu')
    step = tflearn.layers.normalization.batch_normalization(step)
    step = tflearn.layers.core.fully_connected(step, 24, activation='relu')
    step = tflearn.layers.normalization.batch_normalization(step)
    step_low = tflearn.layers.core.fully_connected(step, 32, activation='relu')
    step = step_low
    step = tflearn.layers.normalization.batch_normalization(step)
    step = tflearn.layers.core.fully_connected(step, 64, activation='relu')
    step_high = tflearn.layers.normalization.batch_normalization(step)

    step_high = tf.reshape(step_high, (-1, 1, 1, 64))

    """ small encoder-cnn: """
    feat_e_0 = tflearn.layers.conv.conv_2d(inp_rgb, 64, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_0 = tflearn.layers.conv.max_pool_2d(feat_e_0, [2, 2], [2, 2], padding='VALID')
    feat_e_0 = tf.contrib.layers.layer_norm(feat_e_0)
    #print(feat_e_0)

    feat_e_1 = tflearn.layers.conv.conv_2d(feat_e_0, 64, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_1 = tflearn.layers.conv.max_pool_2d(feat_e_1, [2, 2], [2, 2], padding='VALID')
    feat_e_1 = tf.contrib.layers.layer_norm(feat_e_1)
    #print(feat_e_1)

    feat_e_2 = tflearn.layers.conv.conv_2d(feat_e_1, 64, (1, 1), strides=1, activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_2 = tflearn.layers.conv.max_pool_2d(feat_e_2, [2, 2], [2, 2], padding='VALID')
    feat_e_2 = tf.contrib.layers.layer_norm(feat_e_2)
    #print(feat_e_2)

    feat_e_3 = tflearn.layers.conv.conv_2d(feat_e_2, 256, (1, 1), strides=(1, 1), activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_3 = tflearn.layers.conv.max_pool_2d(feat_e_3, [2, 2], [4, 4], padding='SAME')
    feat_e_3 = tf.contrib.layers.layer_norm(feat_e_3)
    #print(feat_e_3)

    feat_e_4 = tflearn.layers.conv.conv_2d(feat_e_3, 496, (1, 1), strides=(1, 1), activation='relu', weight_decay=1e-5,
                                           regularizer='L2')
    feat_e_4 = tflearn.layers.conv.max_pool_2d(feat_e_4, [2, 2], [6, 6], padding='VALID')
    feat_e_4 = tf.contrib.layers.layer_norm(feat_e_4)
    #print(feat_e_4)  # --> (?,1,1,496)


    #print("step_high", step_high)
    #print("step_low", step_low)

    feat = tf.concat([feat_e_4, step_high], axis=-1)  # feat_C4 = tf.concat([feat_C4, step_high], axis=-1)
    #print("feat", feat)  # --> (1,1,560)

    """ small decoder-cnn: """
    feat = tf.reshape(feat, (-1, 7, 10, 8))

    feat = tflearn.layers.conv.conv_2d_transpose(feat, 496, [3, 2], [15, 20], strides=2, activation='relu',
                                                 weight_decay=1e-5, regularizer='L2', padding="valid")
    feat = tf.contrib.layers.layer_norm(feat)
    #print("feat", feat)

    feat = tflearn.layers.conv.conv_2d_transpose(feat, 496, [2, 2], [30, 40], strides=2, activation='relu',
                                                 weight_decay=1e-5, regularizer='L2')  # padding same
    feat = tf.contrib.layers.layer_norm(feat)
    #print("feat", feat)

    feat = tflearn.layers.conv.conv_2d_transpose(feat, 256, [3, 3], [60, 80], strides=2, activation='relu',
                                                 weight_decay=1e-5, regularizer='L2')  # padding same
    feat = tf.contrib.layers.layer_norm(feat)
    #print("feat", feat)

    feat = tflearn.layers.conv.conv_2d_transpose(feat, 1, [3, 3], [120, 160], strides=2, activation='linear',
                                                 weight_decay=1e-5, regularizer='L2')  # padding same
    feat = tf.contrib.layers.layer_norm(feat)
    #print("feat", feat)

    return feat



def main():
    # create tensorflow session
    sess = tf.Session()

    try:
        args = get_args()
        config = process_config(args.config)

        config.old_tfrecords = args.old_tfrecords
        config.normalize_data = args.normalize_data

    except Exception as e:
        print("An error occurred during processing the configuration file")
        print(e)
        exit(0)


    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.config_file_dir])

    # create your data generator
    train_data = DataGenerator(config, sess, train=True)
    test_data = DataGenerator(config, sess, train=False)

    print("using {} rollout steps".format(config.n_rollouts))

    inp_rgb = tf.placeholder("float", [None, 120, 160, 7])
    control = tf.placeholder("float", [None, 6])
    gt_seg = tf.placeholder("float", [None, 120, 160])

    pred = cnnmodel(inp_rgb, control)

    predictions = tf.reshape(pred, [-1, pred.get_shape()[1] * pred.get_shape()[2]])
    labels = tf.reshape(gt_seg, [-1, gt_seg.get_shape()[1] * gt_seg.get_shape()[2]])

    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions))
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss, global_step=global_step_tensor)

    with tf.variable_scope('cur_epoch'):
        cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
        increment_cur_epoch_tensor = tf.assign(cur_epoch_tensor, cur_epoch_tensor + 1)

    with tf.variable_scope('global_step'):
        cur_batch_tensor = tf.Variable(0, trainable=False, name='cur_batch')
        increment_cur_batch_tensor = tf.assign(cur_batch_tensor, cur_batch_tensor+1)

    next_element_train = train_data.get_next_batch()
    next_element_test = test_data.get_next_batch()

    saver = tf.train.Saver(max_to_keep=config.max_checkpoints_to_keep)

    latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        print("Model loaded")

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    def _process_rollouts(feature, train=True):
        gt_merged_seg_rollout_batch = []
        input_merged_images_rollout_batch = []
        gripper_pos_vel_rollout_batch = []
        for step in range(config.n_rollouts - 1):
            if step < feature["unpadded_experiment_length"]:
                obj_segments = feature["object_segments"][step]
                """ transform (3,120,160,7) into (1,120,160,7) by merging the rgb,depth and seg masks """
                input_merged_images = create_full_images_of_object_masks(obj_segments)

                obj_segments_gt = feature["object_segments"][step + 1]
                gt_merged_seg = create_full_images_of_object_masks(obj_segments_gt)[:, :, 3]

                gripper_pos = feature["gripperpos"][step + 1]
                gripper_vel = feature["grippervel"][step + 1]
                gripper_pos_vel = np.concatenate([gripper_pos, gripper_vel])

                gt_merged_seg_rollout_batch.append(gt_merged_seg)
                input_merged_images_rollout_batch.append(input_merged_images)
                gripper_pos_vel_rollout_batch.append(gripper_pos_vel)

        retrn = sess.run([optimizer, loss, pred], feed_dict={inp_rgb: input_merged_images_rollout_batch,
                                                             control: gripper_pos_vel_rollout_batch,
                                                             gt_seg: gt_merged_seg_rollout_batch})
        if not train:
            """ sigmoid cross entropy runs logits through sigmoid but only during train time """
            seg_data = sigmoid(retrn[2])
            seg_data[seg_data >= 0.5] = 1.0
            seg_data[seg_data < 0.5] = 0.0
            return retrn[1], seg_data, gt_merged_seg_rollout_batch

        return retrn[1], retrn[2]

    for cur_epoch in range(cur_epoch_tensor.eval(sess), config.n_epochs + 1, 1):
        while True:
            try:

                features = sess.run(next_element_train)
                features = convert_dict_to_list_subdicts(features, config.train_batch_size)
                loss_batch = []
                sess.run(increment_cur_batch_tensor)
                for _ in range(config.train_batch_size):
                    for feature in features:
                        loss_train, _ = _process_rollouts(feature)
                        loss_batch.append([loss_train])

                cur_batch_it = cur_batch_tensor.eval(sess)
                loss_avg = np.mean(loss_batch)
                print('train loss batch {0:} is: {1:.4f}'.format(cur_batch_it, loss_avg))

                if cur_batch_it % config.test_interval == 1:
                    loss_batch_test = []

                    print("Executing test batch")
                    features_idx = 0  # always take first element for testing
                    features = sess.run(next_element_test)
                    features = convert_dict_to_list_subdicts(features, config.test_batch_size)
                    seg_img_batch = []

                    loss_valid, data, gt_seg_data = _process_rollouts(features[features_idx], train=False)

                    seg_img_batch.append(data)
                    loss_batch_test.append(loss_valid)
                    loss_avg_test = np.mean(loss_batch_test)
                    print('test loss batch {0:} is: {1:.4f}'.format(cur_batch_it, loss_avg_test))
                    """ create gif here """
                    create_seg_gif(features, features_idx, config, seg_img_batch[0], gt_seg_data, dir_name="tests_during_training", cur_batch_it=cur_batch_it)

                if cur_batch_it % config.model_save_step_interval == 1:
                    print("Saving model...")
                    saver.save(sess, config.checkpoint_dir, global_step_tensor)
                    print("Model saved")


            except tf.errors.OutOfRangeError:
                break

        sess.run(increment_cur_epoch_tensor)


        return None


def create_full_images_of_object_masks(object_segments):
    object_seg_3_obj = object_segments[-3:, :, :, :]
    rgb_all_objects = object_seg_3_obj[0, :, :, :3] + object_seg_3_obj[1, :, :, :3] + object_seg_3_obj[2, :, :, :3]
    seg_all_objects = np.expand_dims(object_seg_3_obj[0, :, :, 3] + object_seg_3_obj[1, :, :, 3] + object_seg_3_obj[2, :, :, 3], axis=3)
    depth_all_objects = object_seg_3_obj[0, :, :, -3:] + object_seg_3_obj[1, :, :, -3:] + object_seg_3_obj[2, :, :, -3:]
    return np.dstack((rgb_all_objects, seg_all_objects, depth_all_objects))


def create_seg_gif(features, features_idx, config, dnn_seg_output, gt_seg_data, dir_name, cur_batch_it):
    prefix = config.exp_name
    dir_path = check_exp_folder_exists_and_create(features=features, features_index=features_idx, prefix=config.exp_name, dir_name=dir_name, cur_batch_it=cur_batch_it)

    # results already exist
    if not dir_path:
        return None

    unpad_exp_length = features[features_idx]["unpadded_experiment_length"]

    target_summaries_dict_seg = {
        prefix + '_target_seg_exp_id_{}_batch_{}_object_0'.format(features[features_idx]['experiment_id'],
                                                                  cur_batch_it): np.expand_dims(np.asarray(gt_seg_data), axis=4)}

    predicted_summaries_dict_seg = {
        prefix + '_predicted_seg_exp_id_{}_batch_{}_object_0'.format(int(features[features_idx]['experiment_id']),
                                                                      cur_batch_it): dnn_seg_output}


    image_dicts = {**target_summaries_dict_seg, **predicted_summaries_dict_seg}
    save_to_gif_from_dict(image_dicts, destination_path=dir_path, unpad_exp_length=unpad_exp_length, only_seg=True)

if __name__ == '__main__':
    main()


