import tensorflow as tf
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from utils.utils import chunks, get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir, convert_batch_to_list
from data_prep.segmentation import get_segments_from_all_experiments, get_segments_from_experiment_step
from data_prep.segmentation import get_number_of_segment


NUM_SEQUENCES_PER_BATCH = 10

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def add_to_dict(dct, data, i, key):
    if i in dct.keys():
        lst = dct[i]
    else:
        lst = []
    lst.append(data[key])
    dct[i] = lst
    return dct

def create_tfrecords_from_dir(source_path, dest_path, name="train", discard_varying_number_object_experiments=True):
    """

    :param source_path:
    :param dest_path:
    :param name:
    :return:
    """
    file_paths = get_all_experiment_file_paths_from_dir(source_path)
    filenames_split = list(chunks(file_paths, NUM_SEQUENCES_PER_BATCH))


    for i, batch in enumerate(filenames_split, 1):
        loaded_batch = load_all_experiments_from_dir(batch)

        filename = os.path.join(dest_path, name + str(i) + '_of_' + str(len(filenames_split)) + '.tfrecords')
        print('Writing', filename)

        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        feature = {}
        identifier = "_object_full_seg_rgb"

        with tf.python_io.TFRecordWriter(path=filename, options=options) as writer:
            for experiment in loaded_batch.values():

                if not experiment:
                    continue

                objects_segments = {}
                gripperpos = []
                objpos = []
                objvel = []
                img = []
                seg = []

                if discard_varying_number_object_experiments:
                    skip = check_if_skip(experiment)
                    if skip:
                        continue

                number_of_total_objects = get_number_of_segment(experiment[0]['seg'])
                experiment_id = experiment[0]['experiment_id']

                objects_segments, gripperpos, objpos, objvel, img, seg = add_experiment_data_to_lists(
                                                            experiment, objects_segments, gripperpos, objpos, objvel, img, seg,
                                                            identifier, seg_only_for_initialization=True)

                # can't store a list of lists as tfrecords, therefore concatenate all object lists (which each contain an entire
                # trajectory (experiment length) and store trajectory length to reconstruct initial per-object list
                # final list has length number_objects * experiment_length, object 0 has elements 0..exp_length etc.
                objects_segments = [list(objects_segments[i]) for i in objects_segments.keys()]
                objects_segments = [lst.tobytes() for objct in objects_segments for lst in objct] # todo: check if converts back

                gripperpos = [array.tobytes() for array in gripperpos] # convert back with np.frombuffer

                # todo: check why objpos and objvel cannot be resolved
                feature["experiment_length"] = len(experiment.keys())
                feature['img'] = _bytes_feature(img)
                feature['seg'] = _bytes_feature(seg)
                feature['object_segments'] = _bytes_feature(objects_segments)
                feature['gripperpos'] = _bytes_feature(gripperpos)
                #feature['objpos'] = _float_feature(objpos)
                #feature['objvel'] = _float_feature(objvel)
                feature['experiment_id'] = _bytes_feature(experiment_id.to_string())
                feature['n_total_objects'] = number_of_total_objects
                feature['n_manipulable_objects'] = number_of_total_objects - 2 # container and gripper subtracted (background is removed)

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def check_if_skip(experiment):
    check_lst = []
    for step in experiment.values():
        check_lst.append(get_number_of_segment(step["seg"]))

    if len(np.unique(check_lst)) is 1:
        # if number of objects vary and should be discarded, go to next experiment
        return False
    return True

def add_experiment_data_to_lists(experiment, objects_segments, gripperpos, objpos, objvel, img, seg, identifier,
                                 seg_only_for_initialization=True):

    for j, trajectory_step in enumerate(experiment.values()):
        segments = get_segments_from_experiment_step([trajectory_step['img'], trajectory_step['seg']])
        if not seg_only_for_initialization:
            for k in range(segments['n_segments']):
                # segmentations (single) are object-specific
                objects_segments = add_to_dict(dct=objects_segments, data=segments, i=k, key=str(k) + identifier)
        else:
            objects_segments = add_to_dict(dct=objects_segments, data=segments, i=0, key=str(0) + identifier)

        # img, seg, gripperpos and objvel is object-unspecific
        seg.append(trajectory_step['seg'].tobytes())
        img.append(trajectory_step['img'].tobytes())
        gripperpos.append(trajectory_step['gripperpos'])
        objpos.append(trajectory_step['objpos'].tolist())
        objvel.append(trajectory_step['objvel'].tolist())

    return objects_segments, gripperpos, objpos, objvel, img, seg

if __name__ == '__main__':
    create_tfrecords_from_dir("../data/source", "../data/destination")
