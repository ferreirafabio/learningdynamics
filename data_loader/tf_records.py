import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.utils import chunks, get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir, convert_batch_to_list
from data_prep.segmentation import get_segments_from_all_experiments



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
        loaded_batch_list = convert_batch_to_list(loaded_batch, ["img", "seg"])
        segments = get_segments_from_all_experiments(loaded_batch_list)

        for experiment in segments:
            for experiment_step in experiment:
                for img in experiment_step.values():
                    plt.imshow(img)
                    plt.show()

        filename = os.path.join(dest_path, name + str(i) + '_of_' + str(len(filenames_split)) + '.tfrecords')
        print('Writing', filename)
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(filename, options=options)

        feature = {}

        for experiment in loaded_batch.values():
            for i, trajectory_step in enumerate(experiment.values()):
                feature['imagecount'] = _int64_feature(i)
                feature['img'] = _bytes_feature(trajectory_step['img'].tostring())
                feature['seg'] = _bytes_feature(trajectory_step['seg'].tostring())


                # todo: add additional segmentation masks
                feature['gripper_pos_x'] = _float_feature(trajectory_step['gripperpos'][0])
                feature['gripper_pos_y'] = _float_feature(trajectory_step['gripperpos'][1])
                feature['gripper_pos_z'] = _float_feature(trajectory_step['gripperpos'][2])
                # todo: check why objpos and objvel cannot be resolved
                feature['obj_positions'] = _float_feature(trajectory_step['objpos'])
                feature['obj_velocities'] = _float_feature(trajectory_step['objvel'])

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        if writer is not None:
            writer.close()


if __name__ == '__main__':
    create_tfrecords_from_dir("../data/source", "../data/destination")
