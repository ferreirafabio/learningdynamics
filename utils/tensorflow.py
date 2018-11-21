import numpy as np

def create_predicted_summary_dicts(images_seg, images_depth, images_rgb, prefix, features, features_index, cur_batch_it):
    predicted_summaries_dict_seg = {
    prefix + '_predicted_seg_exp_id_{}_batch_{}_object_{}'.format(int(features[features_index]['experiment_id']), cur_batch_it, i): obj for
    i, obj in enumerate(images_seg)}

    predicted_summaries_dict_depth = {
    prefix + '_predicted_depth_exp_id_{}_batch_{}_object_{}'.format(int(features[features_index]['experiment_id']), cur_batch_it, i): obj
    for i, obj in enumerate(images_depth)}

    predicted_summaries_dict_rgb = {
    prefix + '_predicted_rgb_exp_id_{}_batch_{}_object_{}'.format(int(features[features_index]['experiment_id']), cur_batch_it, i): obj for
    i, obj in enumerate(images_rgb)}

    return predicted_summaries_dict_seg, predicted_summaries_dict_depth, predicted_summaries_dict_rgb


def create_target_summary_dicts(prefix, features, features_index, cur_batch_it):
    ''' get the ground truth images for comparison, [-3:] means 'get the last three manipulable objects '''
    n_manipulable_objects = features[features_index]['n_manipulable_objects']
    # shape [exp_length, n_objects, w, h, c] --> shape [n_objects, exp_length, w, h, c] --> split in n_objects lists -->
    # [n_split, n_objects, exp_length, ...]
    lists_obj_segs = np.split(np.swapaxes(features[features_index]['object_segments'], 0, 1)[-n_manipulable_objects:], n_manipulable_objects)

    target_summaries_dict_rgb = {
    prefix + '_target_rgb_exp_id_{}_batch_{}_object_{}'.format(features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(
        lst[..., :3], axis=0) for i, lst in enumerate(lists_obj_segs)}

    target_summaries_dict_seg = {
    prefix + '_target_seg_exp_id_{}_batch_{}_object_{}'.format(features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(
        np.expand_dims(lst[..., 3], axis=4), axis=0) for i, lst in enumerate(lists_obj_segs)}

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
        prefix + '_target_global_depth_exp_id_{}_batch_{}'.format(cur_batch_it, features[features_index]['experiment_id'], cur_batch_it):
            features[features_index]['depth']}

    return target_summaries_dict_rgb, target_summaries_dict_seg, target_summaries_dict_depth, target_summaries_dict_global_img, \
           target_summaries_dict_global_seg, target_summaries_dict_global_depth