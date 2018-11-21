import argparse
import os
import re
import collections
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import tensorflow as tf

from graph_nets import utils_tf
from skimage import img_as_ubyte

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def convert_batch_to_list(batch, fltr):
    """
    Args:
        batch:
        fltr: a list of words specifying the keys to keep in the list
    Returns:

    """
    assert type(batch) == dict
    data = []
    for batch_element in batch.values():
        sublist = []
        for i in batch_element.values():
            sublist.append([v for k, v in i.items() for i in fltr if i in k])
        data.append(sublist)
    return data

def chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]


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



def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

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

def save_to_gif_from_dict(image_dicts, destination_path, fps=10):
    if not isinstance(image_dicts, dict) or image_dicts is None:
        return None

    for file_name, img_data in image_dicts.items():
        if len(img_data.shape) == 4 and img_data.shape[3] == 1:
            ''' segmentation masks '''
            clip = mpy.ImageSequenceClip(list(img_data), fps=fps, ismask=True)
        elif len(img_data.shape) == 4 and img_data.shape[3] == 3:
            if img_data.dtype == np.int16:
                img_data = img_as_ubyte(img_data)
            ''' all 3-channel data (rgb, depth etc.)'''
            clip = mpy.ImageSequenceClip(list(img_data.astype(np.int16)), fps=fps, ismask=False)
        else:
            continue

        clip.write_gif(os.path.join(destination_path, file_name) + ".gif", program="ffmpeg", verbose=False, progress_bar=False)


def convert_dict_to_list(dct):
    """ assumes a dict of subdicts of which each subdict only contains one key containing the desired data """
    lst = []
    for value in dct.values():
        if len(value) > 1:
            lst.append(list(value.values()))
        else:
            element = next(iter(value.values())) # get the first element, assuming the dicts contain only the desired data
            lst.append(element)
    return lst

def convert_dict_to_list_subdicts(dct, length):
    list_of_subdicts = []
    for i in range(length):
        batch_item_dict = {}
        for k, v in dct.items():
            batch_item_dict[k] = v[i]
        list_of_subdicts.append(batch_item_dict)
    return list_of_subdicts


def create_dir(output_dir, dir_name):
    assert(output_dir)
    output_dir = os.path.join(output_dir, dir_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print('Created custom directory:', dir_name)
        return output_dir
    print('Using existing directory:', dir_name)
    return output_dir


def convert_list_of_dicts_to_list_by_concat(lst):
    """ concatenate all entries of the dicts into an ndarray and append them into a total list """
    total_list = []
    for dct in lst:
        sub_list = []
        for v in list(dct.values()):
            sub_list.append(v)
        sub_list = np.concatenate(sub_list)
        total_list.append(sub_list)
    return total_list

def convert_float_image_to_int16_legacy(float_image): #todo: remove wrong (65k vs 255) conversion when creating new tfrecords
    dt = float_image.dtype
    float_image = float_image.astype(dt) / float_image.max()
    float_image = 255 * float_image
    return float_image.astype(np.int16)

def convert_image_to_int16(float_image):
    dt = float_image.dtype
    float_image = float_image.astype(dt) / float_image.max()
    float_image = 65536 * float_image
    return float_image.astype(np.int16)

def fix_conversion_error(img): #todo: remove wrong (65k vs 255) conversion when creating new tfrecords
    info = np.iinfo(img.dtype)
    rescaled_img = img.astype(np.float32) / info.max
    rescaled_img = 255 * rescaled_img
    return rescaled_img.astype(np.float32)

def get_number_of_total_samples(tf_records_filenames, options=None):
    c = 0
    for fn in tf_records_filenames:
        for _ in tf.python_io.tf_record_iterator(fn, options=options):
            c += 1
    return c

def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


def get_all_images_from_gn_output(outputs, depth=True):
    images_rgb = []
    images_seg = []
    images_depth = []
    n_objects = np.shape(outputs[0][0])[0]
    if depth:
        img_shape = (120, 160, 7)
    else:
        img_shape = (120, 160, 4)

    for n in range(n_objects):
        rgb = []
        seg = []
        depth_lst = []
        for data_t in outputs:
            image = data_t[0][n][:-6].reshape(img_shape)  # always get the n node features without pos+vel
            rgb.append(image[:, :, :3])
            seg.append(np.expand_dims(image[:, :, 3], axis=2))
            if depth:
                depth_lst.append(image[:, :, -3:])
        images_rgb.append(np.stack(rgb))
        images_seg.append(np.stack(seg))
        if depth:
            images_depth.append(np.stack(depth_lst))
    # todo: possibly expand_dims before stacking since (exp_length, w, h, c) might become (w,h,c) if exp_length = 1
    return images_rgb, images_seg, images_depth


def get_pos_ndarray_from_output(output_for_summary):
    """ returns a position vector from a single step output, example shape: (exp_length,n_objects,3) for 3 = x,y,z dimension"""
    n_objects = np.shape(output_for_summary[0][0][0])[0]
    pos_lst = []
    for data_t in output_for_summary[0]:
        pos_t = []
        for n in range(n_objects):
            pos_object = data_t[0][n][-3:]
            pos_t.append(pos_object)
        pos_lst.append(np.stack(pos_t))

    return pos_lst





if __name__ == '__main__':
    source_path = "../data/source"
    exp_number = 5
    dest_path = os.path.join(source_path, str(exp_number))

    image_data = get_experiment_image_data_from_dir(source_path=source_path, experiment_id=exp_number, data_type="seg", as_dict=False)
    save_image_data_to_disk(image_data, dest_path, img_type="rgb")
    all_image_data = get_all_experiment_image_data_from_dir(source_path, data_type=["rgb", "seg"])

