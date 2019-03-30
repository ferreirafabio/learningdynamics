import collections
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, animation as animation
from moviepy import editor as mpy
from skimage import img_as_ubyte, img_as_int, img_as_uint

from eval.AnimateLatentData import AnimateLatentData
from utils.conversions import convert_dict_to_list


def get_all_experiment_file_paths_from_dir(source_path, file_type=".npz", order=True):
    assert os.path.exists(source_path)

    file_paths = {}
    for root, dirs, files in os.walk(source_path):

        trial_idx = os.path.basename(root)

        files = [i for i in files if file_type in i]

        # make sure array is not empty
        if files:
            dct = {}
            file_paths[trial_idx] = dct

        for file in files:
            if file.endswith(file_type):
                n = re.search("\d+", file).group(0)
                if n in dct.keys():
                    lst = dct[n]
                    lst.append(os.path.join(root, file))
                    dct[n] = lst
                else:
                    lst = []
                    lst.append(os.path.join(root, file))
                    dct[n] = lst


    # sort inner and outer dicts to avoid conflicts in tfrecord generation
    for i in file_paths.keys():
        file_paths[i] = collections.OrderedDict(sorted(file_paths[i].items(), key=lambda x: chr(int(x[0]))))

    file_paths = collections.OrderedDict(sorted(file_paths.items(), key=lambda x: chr(int(x[0]))))

    return [list(file_paths[i].values()) for i in file_paths.keys()]


def load_all_experiments_from_dir(data_list):
    """
    Returns a dict of dicts containing the available attribute information about an experiment, e.g.
    all_experiments
        - experiment 0
            - trajectory sample t=0
                - attributes
            - trajectory sample t=1
                - attributes (e.g. img, segmented image, gripper position, ...)
                etc.

    Args:
        data_list: a collection of lists which again contain lists of directory paths (either relative or absolute), e.g.
            00 = {list}
                00 = {list}
                    0 = {str} '../data/source/0/0gripperpos.npz'
                    1 = {str} '../data/source/0/0objpos.npz'
                    etc.
     """
    all_experiments = {}
    for i, batch_element in enumerate(data_list):
        experiment = load_data_from_list_of_paths(batch_element)
        all_experiments[i] = experiment
    return all_experiments


def load_data_from_list_of_paths(batch_element):
    """
    loads all the data from a single experiment which is represented by a list of paths
    Args:
        batch_element: a list of lists while every list contains a path as a string (can be relative or absolute)
        get_path: a boolean flag indicating whether the path given as input should also be contained in the returned sub-dicts
    Returns:
        A dictionary containing sub-dictionaries while every sub-dict contains the loaded data as ndarray
    """
    experiment = {}
    for j, trajectory in enumerate(batch_element):
        trajectory_step_data = {}
        for traj in trajectory:
            with np.load(traj) as fhandle:
                key = list(fhandle.keys())[0]
                data = fhandle[key]
                # save some space:
                if key in ['img', 'seg']:
                    data = data.astype(np.uint8)
                trajectory_step_data[key] = data
                trajectory_step_data['experiment_id'] = get_dir_name(traj)
        experiment[j] = trajectory_step_data
    return experiment


def get_dir_name(path):
    return os.path.basename(os.path.dirname(path))


def get_all_experiment_image_data_from_dir(source_path, data_type="rgb"):
    all_paths = get_all_experiment_file_paths_from_dir(source_path=source_path)
    image_data = []
    for path in all_paths:
        dct = load_images_of_experiment(path, data_type)
        image_data.append(dct)
    return image_data


def get_experiment_image_data_from_dir(source_path, experiment_id, data_type="seg", as_dict=False):
    """
    loads and returns all the image data from a single experiment. Expects the following folder structure:
        folder 'experiment number'
            0rgb.npz
            1rgb.npz
            etc.

    Args:
        source_path: the root directory of all the experiments
        experiment_id: integer indicating the number of the experiment to be used
        data_type: keyword used to identify the type of data (e.g. either 0rgb.npz or seg.npz), can also be a list, e.g. ['seg', 'rgb']

    Returns:
        A list of m ndarrays representing the single images while is m is the number of images in the experiment
    """
    allowed_types = ['seg', 'rgb']

    if type(data_type) == list:
        assert all([i in allowed_types for i in data_type])
    else:
        assert data_type in ['seg', 'rgb']

    all_paths = get_all_experiment_file_paths_from_dir(source_path)
    try:
        experiment_paths = all_paths[experiment_id]
    except:
        print("no data found under the specified number")
        return

    return load_images_of_experiment(experiment_paths, data_type, as_dict=as_dict)


def load_images_of_experiment(experiment, data_type, as_dict=False):
    """
    loads images from pure paths for an entire experiment
    Args:
        experiment: a list of lists where each list contains paths as strings
        data_type: keyword that identifies the type of files to be loaded into a dict
    Returns:
        a list of ndarrays containing the images
    """
    if type(data_type) != list:
        data_type = [data_type]

    image_data_paths = []
    for experiment_step in experiment:
        image_data_paths.append([path for path in experiment_step for i in data_type if i in path])

    images_dict = load_data_from_list_of_paths(image_data_paths)

    if as_dict:
        return images_dict
    else:
        return convert_dict_to_list(images_dict)


def save_image_data_to_disk(image_data, destination_path, store_gif=True, img_type="seg"):
    assert destination_path is not None

    output_dir = create_dir(destination_path, "images_" + img_type)
    for i, img in enumerate(image_data):
        img = img[0].astype(np.uint8)
        plt.imsave(output_dir + "/" + img_type + str(i).zfill(4), img)

    if store_gif:
        output_dir = create_dir(destination_path, "images" + '_' + img_type)
        clip = mpy.ImageSequenceClip(output_dir, fps=5, with_mask=False).to_RGB()
        clip.write_gif(os.path.join(output_dir, 'sequence_' + img_type + '.gif'), program='ffmpeg')

def _normalize_if_necessary(img_data):
    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        if np.min(img_data) < -1.0 or np.max(img_data) > 1.0:
            ''' normalize [0, 1]'''
            return (img_data - np.min(img_data)) / np.ptp(img_data)
            # img_data = 2*(img_data - np.min(img_data))/np.ptp(img_data)-1
    return img_data


def save_to_gif_from_dict(image_dicts, destination_path, fps=10, use_moviepy=False, overlay_images=True):
    if not isinstance(image_dicts, dict) or image_dicts is None:
        return None

    for file_name, img_data in image_dicts.items():
        img_data = _normalize_if_necessary(img_data)

        img_data_uint = img_as_ubyte(img_data)


        object_id = file_name.split('_')[-2:]
        object_id = object_id[0] + '_' + object_id[1]

        key_seg = [k for k in image_dicts.keys() if object_id in k and 'seg' in k and 'target' in k][0]
        key_rgb = [k for k in image_dicts.keys() if object_id in k and 'rgb' in k and 'target' in k][0]
        key_depth = [k for k in image_dicts.keys() if object_id in k and 'depth' in k and 'target' in k][0]

        if len(img_data_uint.shape) == 4 and img_data_uint.shape[3] == 1:
            ''' segmentation masks '''
            if use_moviepy:
                clip = mpy.ImageSequenceClip(list(img_data_uint), fps=fps, ismask=True)
            else:
                fig = plt.figure(frameon=False)
                dpi = fig.get_dpi()
                fig.set_size_inches(320.0/float(dpi), 240.0/float(dpi))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                fig.add_axes(ax)
                ims = []

                for i in range(img_data_uint.shape[0]+1):  # +1 because of the extra init image
                    if overlay_images and "predicted" in file_name:
                        # treat initial image different: add ground truth init image to the prediction
                        if i == 0:
                            im1 = plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key_seg][0, :, :, 0])), animated=True, interpolation='none')
                            # take from gt data
                            im2 = plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key_seg][0, :, :, 0])), alpha=0.7, animated=True, interpolation='none')
                            im = [im2, im1]

                        else:
                            im1 = plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key_seg][i, :, :, 0])),animated=True, interpolation='none')
                            # take from GN output, use i-1 since GN output has one less than gt data
                            im2 = plt.imshow(img_data_uint[i-1, :, :, 0], alpha=0.7, animated=True, interpolation='none')
                            im = [im2, im1]
                    else:
                        if i < img_data_uint.shape[0]:
                            im = [plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key_seg][i, :, :, 0])), animated=True, interpolation='none')]
                    ims.append(im)
                clip = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)

        elif len(img_data_uint.shape) == 4 and img_data_uint.shape[3] == 3:
            ''' all 3-channel data (rgb, depth etc.)'''
            if use_moviepy:
                clip = mpy.ImageSequenceClip(list(img_data_uint), fps=fps, ismask=False)
            else:
                fig = plt.figure(frameon=False)
                dpi = fig.get_dpi()
                fig.set_size_inches(320.0/float(dpi), 240.0/float(dpi))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ims = []

                if "depth" in file_name:
                    key = key_depth
                else:
                    key = key_rgb

                for i in range(img_data_uint.shape[0]+1):  # +1 because of the extra init image
                    if overlay_images and "predicted" in file_name:
                        # treat initial image different: add ground truth init image to the prediction
                        if i == 0:
                            im1 = plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key][0, :, :, :])), animated=True, interpolation='none')
                            # take from gt data
                            im2 = plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key][0, :, :, :])), alpha=0.7, animated=True, interpolation='none')
                            im = [im2, im1]

                        else:
                            im1 = plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key][i, :, :, :])),animated=True, interpolation='none')
                            # take from GN output, use i-1 since GN output has one less than gt data
                            im2 = plt.imshow(img_data_uint[i-1, :, :, :], alpha=0.7, animated=True, interpolation='none')
                            im = [im2, im1]
                    else:
                        if i < img_data_uint.shape[0]:
                            im = [plt.imshow(img_as_ubyte(_normalize_if_necessary(image_dicts[key][i, :, :, :])), animated=True, interpolation='none')]

                    ims.append(im)
                clip = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)
        else:
            continue

        name = os.path.join(destination_path, file_name)
        if use_moviepy:
            name += ".gif"
            clip.write_gif(name, program="ffmpeg", verbose=False, progress_bar=False)
        else:
            if overlay_images and "predicted" in file_name:
                name += "_overlay" + ".gif"
                clip.save(name, writer='imagemagick')
            else:
                name += ".gif"
                clip.save(name, writer='imagemagick')


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


def export_summary_images(config, summaries_dict_images, dir_path, overlay_images=True):
    save_to_gif_from_dict(image_dicts=summaries_dict_images, destination_path=dir_path, fps=config.n_rollouts, overlay_images=overlay_images)


def export_latent_df(df, dir_path):
    path_pkl = os.path.join(dir_path, "obj_pos_vel_dataframe.pkl")
    df.to_pickle(path_pkl)
    path_csv = os.path.join(dir_path, "obj_pos_vel_dataframe.csv")
    df.to_csv(path_csv)


def create_latent_images(df, features, features_index, dir_path, config, prefix, cur_batch_it):
    """ creates the images corresponding to the latent space such as velocity or position -- currently only implemented for position
        if a directory exists, only the image_dict is generated for summary, otherwise the dict is generated and the images exported
     """
    n_objects = features[features_index]['n_manipulable_objects']
    experiment_id = int(features[features_index]['experiment_id'])

    image_dict = {}
    for i in range(n_objects):
        identifier_gt = "{}_obj_gt_pos".format(i)
        identifier_pred = "{}_obj_pred_pos".format(i)

        animate = AnimateLatentData(df=df, identifier1=identifier_gt, identifier2=identifier_pred, n_rollouts=config.n_rollouts)
        title = 'Ground truth vs predicted centroid position of object {}'.format(i)

        path_3d = os.path.join(dir_path, "3d_obj_pos_object_" + str(i) + ".gif")
        path_2d = os.path.join(dir_path, "2d_obj_pos_object_" + str(i) + ".gif")
        fig_as_array_3d = animate.save_3d_plot(title=title, output_dir=path_3d)
        fig_as_array_2d = animate.save_2d_plot(title=title, output_dir=path_2d)

        image_dict[prefix + "_3d_obj_pos_exp_id_{}_batch_{}_object_{}".format(experiment_id, cur_batch_it, i)] = np.expand_dims(fig_as_array_3d, 0)
        image_dict[prefix + "_2d_obj_pos_exp_id_{}_batch_{}_object_{}".format(experiment_id, cur_batch_it, i)] = np.expand_dims(fig_as_array_2d, 0)

    return image_dict
