

def create_pred_summary_dict(images_seg, images_depth, images_rgb, prefix, features, features_index, cur_batch_it):
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