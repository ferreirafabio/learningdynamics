import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utils.utils import chunks, get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir, convert_batch_to_list
from data_prep.segmentation import get_segments_from_all_experiments, get_segments_from_experiment_step



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


def create_tfrecords_from_dir(source_path, dest_path, name="train"):
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
                objects_segments = {}
                gripperpos = {}
                objvel = {}
                img = {}
                seg = {}

                for j, trajectory_step in enumerate(experiment.values()):
                    segments = get_segments_from_experiment_step([trajectory_step['img'], trajectory_step['seg']])
                    for k in range(segments['n_segments']):

                        # segmentations (single) are object-specific
                        if k in objects_segments.keys():
                            lst = objects_segments[k]
                        else:
                            lst = []

                        lst.append(segments[str(k) + identifier])
                        objects_segments[k] = lst

                    # img, seg, gripperpos and objvel is object-unspecific
                    if j in seg.keys():
                        lst = seg[j]
                    else:
                        lst = []
                    lst.append(trajectory_step['seg'])
                    seg[j] = lst

                    if j in img.keys():
                        lst = img[j]
                    else:
                        lst = []
                    lst.append(trajectory_step['img'])
                    img[j] = lst

                    if j in gripperpos.keys():
                        lst = gripperpos[k]
                    else:
                        lst = []
                    lst.append(trajectory_step['gripperpos'])
                    gripperpos[j] = lst

                    if j in objvel.keys():
                        lst = objvel[j]
                    else:
                        lst = []
                    lst.append(trajectory_step['objvel'])
                    objvel[j] = lst

                    # todo: convert dicts to lists and assign to feature dict




                # todo: add additional segmentation masks
                #feature['gripper_pos_x'] = _float_feature(trajectory_step['gripperpos'][0])
                #feature['gripper_pos_y'] = _float_feature(trajectory_step['gripperpos'][1])
                #feature['gripper_pos_z'] = _float_feature(trajectory_step['gripperpos'][2])
                # todo: check why objpos and objvel cannot be resolved
                #feature['obj_positions'] = _float_feature(trajectory_step['objpos'])
                #feature['obj_velocities'] = _float_feature(trajectory_step['objvel'])


                feature['img'] = _bytes_feature(segments['full_rgb'].tostring())
                feature['seg'] = _bytes_feature(segments['full_seg'].tostring())
                feature['seg'] = _bytes_feature(segments['full_seg'].tostring())
                feature['experiment_id'] = _int64_feature(i)
                feature['segments'] = _bytes_feature(segments_experiment)
                feature['experirment_length'] = _int64_feature()
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())



if __name__ == '__main__':
    create_tfrecords_from_dir("../data/source", "../data/destination")
