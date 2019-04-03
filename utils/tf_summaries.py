import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from utils.utils import get_images_from_gn_output, get_latent_from_gn_output, check_exp_folder_exists_and_create
from utils.io import export_summary_images, export_latent_df, create_latent_images
from utils.math_ops import normalize_df_column, normalize_df




def create_predicted_summary_dicts(images_seg, images_depth, images_rgb, prefix, features, features_index, cur_batch_it, config):
    predicted_summaries_dict_seg = {
    prefix + '_predicted_seg_exp_id_{}_batch_{}_object_{}'.format(int(features[features_index]['experiment_id']), cur_batch_it, i):
        obj for i, obj in enumerate(images_seg)}

    #if not config.normalize_data else img_as_int(obj)

    predicted_summaries_dict_depth = {
    prefix + '_predicted_depth_exp_id_{}_batch_{}_object_{}'.format(int(features[features_index]['experiment_id']), cur_batch_it, i):
        obj for i, obj in enumerate(images_depth)}


    predicted_summaries_dict_rgb = {
    prefix + '_predicted_rgb_exp_id_{}_batch_{}_object_{}'.format(int(features[features_index]['experiment_id']), cur_batch_it, i):
        obj for i, obj in enumerate(images_rgb)}

    return predicted_summaries_dict_seg, predicted_summaries_dict_depth, predicted_summaries_dict_rgb


def create_target_summary_dicts(prefix, features, features_index, cur_batch_it, config):
    ''' get the ground truth images for comparison, [-3:] means 'get the last three manipulable objects '''
    n_manipulable_objects = features[features_index]['n_manipulable_objects']
    # shape [exp_length, n_objects, w, h, c] --> shape [n_objects, exp_length, w, h, c] --> split in n_objects lists -->
    # [n_split, n_objects, exp_length, ...]
    lists_obj_segs = np.split(np.swapaxes(features[features_index]['object_segments'], 0, 1)[-n_manipulable_objects:], n_manipulable_objects)

    target_summaries_dict_rgb = {
    prefix + '_target_rgb_exp_id_{}_batch_{}_object_{}'.format(features[features_index]['experiment_id'], cur_batch_it, i):
        np.squeeze(lst[..., :3], axis=0) for i, lst in enumerate(lists_obj_segs)}

    target_summaries_dict_seg = {
    prefix + '_target_seg_exp_id_{}_batch_{}_object_{}'.format(features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(
        np.expand_dims(lst[..., 4], axis=4), axis=0) for i, lst in enumerate(lists_obj_segs)}

    target_summaries_dict_depth = {
    prefix + '_target_depth_exp_id_{}_batch_{}_object_{}'.format(features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(
        lst[..., -3:], axis=0) for i, lst in enumerate(lists_obj_segs)}

    target_summaries_dict_global_img = {
        prefix + '_target_global_img_exp_id_{}_batch_{}'.format(features[features_index]['experiment_id'], cur_batch_it):
            features[features_index]['img']}

    target_summaries_dict_global_seg = {
        prefix + '_target_global_seg_exp_id_{}_batch_{}'.format(features[features_index]['experiment_id'], cur_batch_it): np.expand_dims(
            features[features_index]['seg'], axis=4)}

    target_summaries_dict_global_depth = {
        prefix + '_target_global_depth_exp_id_{}_batch_{}'.format(features[features_index]['experiment_id'], cur_batch_it):
            features[features_index]['depth']}

    return target_summaries_dict_rgb, target_summaries_dict_seg, target_summaries_dict_depth, target_summaries_dict_global_img, \
           target_summaries_dict_global_seg, target_summaries_dict_global_depth


def create_image_summary(output_for_summary, config, prefix, features, cur_batch_it):
    ''' returns n lists, each having an ndarray of shape (exp_length, w, h, c)  while n = number of objects '''
    images_rgb, images_seg, images_depth = get_images_from_gn_output(output_for_summary[0], config.depth_data_provided)
    features_index = output_for_summary[1]  # assumes outside caller uses for loop to iterate over outputs --> use always first index

    predicted_summaries_dict_seg, predicted_summaries_dict_depth, predicted_summaries_dict_rgb = create_predicted_summary_dicts(
        images_seg, images_depth, images_rgb, prefix=prefix, features=features, features_index=features_index, cur_batch_it=cur_batch_it,
        config=config)


    target_summaries_dict_rgb, target_summaries_dict_seg, target_summaries_dict_depth, target_summaries_dict_global_img, \
    target_summaries_dict_global_seg, target_summaries_dict_global_depth = create_target_summary_dicts(
        prefix=prefix, features=features, features_index=features_index, cur_batch_it=cur_batch_it, config=config)

    summaries_dict_images = {**predicted_summaries_dict_rgb, **predicted_summaries_dict_seg, **predicted_summaries_dict_depth,
                             **target_summaries_dict_rgb, **target_summaries_dict_seg, **target_summaries_dict_depth,
                             **target_summaries_dict_global_img, **target_summaries_dict_global_seg, **target_summaries_dict_global_depth}


    return summaries_dict_images, features_index


def create_latent_data_df(config, output_for_summary, gt_features, unpad_exp_length):
    """ creates a pandas dataframe with rows = timesteps (rollouts) and as columns the predictions / ground truths
     of velocities and columns, e.g.
        0_obj_pred_pos, 0_obj_gt_pos, 1_obj_pred_pos, 1_obj_gt_pos, ... , 0_obj_pred_vel, 0_obj_gt_vel, ...
    0   [...], ..., [...]
    1

    the values are unnormalized.

    """
    pos, vel = get_latent_from_gn_output(output_for_summary[0])  # exclude the index

    features_index = output_for_summary[1]
    
    # in the case we have more gt data than predictions:
    if gt_features[features_index]['experiment_length'] != len(output_for_summary[0]):
        cut_length = len(output_for_summary[0])
        if cut_length > unpad_exp_length:
            cut_length = unpad_exp_length
    else:
        cut_length = gt_features[features_index]['experiment_length']

    pos = [pos_of_obj[:cut_length] for pos_of_obj in pos]
    vel = [vel_of_obj[:cut_length] for vel_of_obj in vel]

    pos_gt, vel_gt = get_latent_target_data(gt_features, features_index, cut_length)

    # pos, vel are predictions given an initial state t=0 --> gt data always has one more sample
    # --> add this sample to predictions
    # if config.initial_pos_vel_known:
    #     for i, pos_of_obj in enumerate(pos):
    #         pos_of_obj.insert(0, pos_gt[i][0])
    #     for i, vel_of_obj in enumerate(vel):
    #         vel_of_obj.insert(0, vel_gt[i][0])
    # else:
    #     for pos_of_obj in pos:
    #         pos_of_obj.insert(0, np.zeros(shape=3))
    #     for vel_of_obj in vel:
    #         vel_of_obj.insert(0, np.zeros(shape=3))

    n_objects = np.shape(output_for_summary[0][0][0])[0]

    """ position header """
    header_pos_pred = [str(i) + "_obj_pred_pos" for i in range(n_objects)]
    header_pos_gt = [str(i) + "_obj_gt_pos" for i in range(n_objects)]
    header_pos = sum(zip(header_pos_gt, header_pos_pred), ())  # alternating list [#0_pred, 0_gt, 1_pred, 1_gt...]

    """ velocity header """
    header_vel_pred = [str(i) + "_obj_pred_vel" for i in range(n_objects)]
    header_vel_gt = [str(i) + "_obj_gt_vel" for i in range(n_objects)]
    header_vel = sum(zip(header_vel_gt, header_vel_pred), ())  # alternating list [0_pred, 0_gt, 1_pred, 1_gt...]

    all_pos = sum(zip(pos_gt, pos), ())  # alternate pos and pos_gt in a list
    all_vel = sum(zip(vel_gt, vel), ())

    all_data = all_pos + all_vel
    all_header = header_pos + header_vel

    df = pd.DataFrame.from_items(zip(all_header, all_data))

    """ testing """
    np.testing.assert_array_equal(df.ix[:,0].tolist(), pos_gt[0])  # check first column
    np.testing.assert_array_equal(df.ix[:,-1].tolist(), vel[-1])  # check last column

    df_normalized = normalize_df(df.ix[:, :6].copy())  # only normalize positions, leave velocities normalized
    df_normalized = df_normalized.join(df.ix[:, -6:])  # append velocities

    def _compute_df_statistics(df, n_objects):
        """ compute statistics of pos """
        for i in range(0, n_objects*2, 2):  # 2: each one column for pred and gt
            column_name = list(df.columns.values)[i] + '-' + list(df.columns.values)[i+1]

            df['mean'+'('+column_name+')'] = [(df.ix[:, i] - df.ix[:, i+1]).mean(axis=0)] * len(df.index)
            df['std' + '(' + column_name + ')'] = [np.std((df.ix[:, i] - df.ix[:, i+1]).tolist(), axis=0)] * len(df.index)

        """ compute statistics of vel """
        for i in range(n_objects * 2, (n_objects * 2)*2, 2):
            column_name = list(df.columns.values)[i] + '-' + list(df.columns.values)[i + 1]
            # compute mean / std and repeat (df.index)-lines to construct a pandas series
            df['mean' + '(' + column_name + ')'] = [(df.ix[:, i] - df.ix[:, i + 1]).mean(axis=0)] * len(df.index)
            df['std' + '(' + column_name + ')'] = [np.std((df.ix[:, i] - df.ix[:, i+1]).tolist(), axis=0)] * len(df.index)

        return df

    df = _compute_df_statistics(df, n_objects)
    df_normalized = _compute_df_statistics(df_normalized, n_objects)

    return df, df_normalized


def generate_results(output, config, prefix, features, cur_batch_it, export_images, export_latent_data, dir_name, reduce_dict=True, overlay_images=True):

    summaries_dict_images, features_index = create_image_summary(output, config=config, prefix=prefix, features=features,
                                                 cur_batch_it=cur_batch_it)

    dir_path = check_exp_folder_exists_and_create(features, features_index, prefix, dir_name, cur_batch_it)

    summaries_pos_dict_images = None

    unpad_exp_length = features[features_index]['unpadded_experiment_length']

    if export_images and dir_path:  # skip if directory exists
        export_summary_images(config=config, summaries_dict_images=summaries_dict_images, dir_path=dir_path, overlay_images=overlay_images)

    if export_latent_data and dir_path:
        """ this will generate a pandas dataframe of unnormalized values. 'create_latent_images' then uses this df, normalizes the values and plots them"""
        df, _ = create_latent_data_df(config, output, gt_features=features, unpad_exp_length=unpad_exp_length)
        export_latent_df(df=df, dir_path=dir_path)

        if export_images:
            summaries_pos_dict_images = create_latent_images(df=df, features=features, features_index=features_index, dir_path=dir_path,
                                                          config=config, prefix=prefix, cur_batch_it=cur_batch_it)

    keys = ["seg"]  # will only yield segmentation images
    if reduce_dict:
        summaries_dict_images = {summary_key: summaries_dict_images[summary_key] for summary_key in summaries_dict_images.keys() for k in keys if k in summary_key}

    return summaries_dict_images, summaries_pos_dict_images


def get_latent_target_data(features, features_index, limit=None):
    n_manipulable_objects = features[features_index]['n_manipulable_objects']
    list_obj_pos = np.split(np.swapaxes(features[features_index]['objpos'], 0, 1)[:n_manipulable_objects], n_manipulable_objects)
    list_obj_vel = np.split(np.swapaxes(features[features_index]['objvel'], 0, 1)[:n_manipulable_objects], n_manipulable_objects)
    list_obj_pos = [list(np.squeeze(i))[:limit] for i in list_obj_pos]  # remove 1 dim and transform list of ndarray to list of lists
    list_obj_vel = [list(np.squeeze(i))[:limit] for i in list_obj_vel]

    return list_obj_pos, list_obj_vel
