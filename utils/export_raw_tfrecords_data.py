import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.singulation_trainer import SingulationTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from pydoc import locate
import numpy as np
import os
from matplotlib import pyplot as plt, animation as animation
from moviepy import editor as mpy
from skimage import img_as_ubyte


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

        config.old_tfrecords = args.old_tfrecords
        config.normalize_data = False


    except Exception as e:
        print("An error occurred during processing the configuration file")
        print(e)
        exit(0)

        # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.config_file_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    #train_data = DataGenerator(config, sess, train=True)


    test_data = DataGenerator(config, sess, train=False)
    next_element = test_data.get_next_batch()
    dir_name = "test"

    while True:
        try:
            features = sess.run(next_element)
            features = convert_dict_to_list_subdicts(features, config.test_batch_size)
            features = features[0]
            summaries = create_target_summary_dicts(features)

            dir_path = check_exp_folder_exists_and_create(features, dir_name)
            if dir_path is not None:
                export_summary_images(config=config, summaries_dict_images=summaries, dir_path=dir_path)

        except tf.errors.OutOfRangeError:
            print("done exporting")
            break



def create_target_summary_dicts(features):
    ''' get the ground truth images for comparison, [-3:] means 'get the last three manipulable objects '''
    n_manipulable_objects = features['n_manipulable_objects']
    # shape [exp_length, n_objects, w, h, c] --> shape [n_objects, exp_length, w, h, c] --> split in n_objects lists -->
    # [n_split, n_objects, exp_length, ...]
    lists_obj_segs = np.split(np.swapaxes(features['object_segments'], 0, 1)[-n_manipulable_objects:], n_manipulable_objects)

    target_summaries_dict_global_img = {
        '_target_global_img_exp_id_{}'.format(features['experiment_id']):
            features['img']}

    target_summaries_dict_global_seg = {
        '_target_global_seg_exp_id_{}'.format(features['experiment_id']): np.expand_dims(
            features['seg'], axis=4)}

    target_summaries_dict_global_depth = {
        '_target_global_depth_exp_id_{}'.format(features['experiment_id']):
            features['depth']}

    return {**target_summaries_dict_global_img, **target_summaries_dict_global_seg, **target_summaries_dict_global_depth}


def check_exp_folder_exists_and_create(features, dir_name):
    exp_id = features['experiment_id']
    if dir_name is not None:
        dir_path, _ = create_dir("../data/tf_records_raw_data", dir_name)
        dir_path, exists = create_dir(dir_path, "{}_{}".format(exp_id, dir_name))
        if exists:
            print("skipping export for exp_id: {} (directory already exists)".format(exp_id))
            return None
    return dir_path


def convert_dict_to_list_subdicts(dct, length):
    list_of_subdicts = []
    for i in range(length):
        batch_item_dict = {}
        for k, v in dct.items():
            batch_item_dict[k] = v[i]
        list_of_subdicts.append(batch_item_dict)
    return list_of_subdicts

def create_dir(output_dir, dir_name, verbose=False):
    exists = False
    assert(output_dir)
    output_dir = os.path.join(output_dir, dir_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print('Created custom directory:', dir_name)
        return output_dir, exists
    exists = True
    if verbose:
        print('Using existing directory:', dir_name)
    return output_dir, exists

def export_summary_images(config, summaries_dict_images, dir_path):
    save_to_gif_from_dict(image_dicts=summaries_dict_images, destination_path=dir_path, fps=config.n_rollouts)


def save_to_gif_from_dict(image_dicts, destination_path, fps=10, use_moviepy=False):
    if not isinstance(image_dicts, dict) or image_dicts is None:
        return None

    for file_name, img_data in image_dicts.items():
        if img_data.dtype == np.float32 or img_data.dtype == np.float64:
            ''' normalize [-1, 1]'''
            img_data = 2*(img_data - np.min(img_data))/np.ptp(img_data)-1

        img_data_uint = img_as_ubyte(img_data)
        if len(img_data_uint.shape) == 4 and img_data_uint.shape[3] == 1:
            if use_moviepy:
                ''' segmentation masks '''
                clip = mpy.ImageSequenceClip(list(img_data_uint), fps=fps, ismask=True)
            else:
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ims = []
                for i in range(img_data_uint.shape[0]):
                    ims.append([plt.imshow(img_data_uint[i,:,:,0], animated=True)])
                clip = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

        elif len(img_data_uint.shape) == 4 and img_data_uint.shape[3] == 3:
            if use_moviepy:
                ''' all 3-channel data (rgb, depth etc.)'''
                clip = mpy.ImageSequenceClip(list(img_data_uint), fps=fps, ismask=False)
            else:
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ims = []
                for i in range(img_data_uint.shape[0]):
                    ims.append([plt.imshow(img_data_uint[i,:,:,:], animated=True)])
                clip = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
        else:
            continue

        name = os.path.join(destination_path, file_name) + ".gif"
        if use_moviepy:
            clip.write_gif(name, program="ffmpeg", verbose=False, progress_bar=False)
        else:
            clip.save(name, writer='imagemagick')



if __name__ == '__main__':
    main()
