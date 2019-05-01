from utils.io import get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir

from utils.io import save_to_gif_from_dict
from utils.io import get_all_experiment_image_data_from_dir, get_experiment_image_data_from_dir, save_image_data_to_disk, create_dir

import numpy as np
import os

def create_merged_gifs_from_numpy(source_path):
    obj_ids = ["_3", "_4", "_5"]
    file_paths = get_all_experiment_file_paths_from_dir(source_path, file_type='.npy')
    loaded_batch = load_all_experiments_from_dir(file_paths, single_object_files=True, obj_ids=obj_ids)

    prefix='baseline_results_lin'


    for exp in loaded_batch.values():
        for obj_id in obj_ids:
            exp_id = exp[0][obj_id]['experiment_id']
            dnn_seg_outputs = []
            for step in exp.values():
                dnn_seg_output = step[obj_id]['seg']
                dnn_seg_output[dnn_seg_output >= 0.5] = 1.0
                dnn_seg_output[dnn_seg_output < 0.5] = 0.0
                dnn_seg_outputs.append(dnn_seg_output)

            gt_seg_data = [step[obj_id]['seg_target'].astype(np.float) for step in exp.values()]

            target_summaries_dict_seg = {
                prefix + '_target_seg_exp_id_{}_obj_id{}'.format(exp_id, obj_id): np.expand_dims(np.asarray(gt_seg_data), axis=4)}

            predicted_summaries_dict_seg = {
                prefix + '_predicted_seg_exp_id_{}_obj_id{}'.format(exp_id, obj_id): np.expand_dims(np.asarray(dnn_seg_outputs), axis=4)}

            image_dicts = {**target_summaries_dict_seg, **predicted_summaries_dict_seg}

            dir_name = "baseline_comparison"
            dir_path = check_exp_folder_exists_and_create(exp_id, prefix, dir_name)


            save_to_gif_from_dict(image_dicts, destination_path=dir_path, unpad_exp_length=0, only_seg=True)


def check_exp_folder_exists_and_create(exp_id, prefix, dir_name):
    if dir_name is not None:
        dir_path, _ = create_dir(os.path.join("../experiments", prefix), dir_name)
        dir_path, exists = create_dir(dir_path, "summary_images_exp_id_{}".format(exp_id))
        if exists:
            print("skipping export for exp_id: {} (directory already exists)".format(exp_id))
            return dir_path
    else:
        dir_path = create_dir(os.path.join("../experiments", prefix), "summary_images_exp_id_{}".format(exp_id))
    return dir_path


if __name__ == '__main__':
    create_merged_gifs_from_numpy(source_path="/juno/u/lins2/GN/neuralnets_condition/saved_results")