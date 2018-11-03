import tensorflow as tf
import numpy as np

seg_only_for_initialization = True

def read_and_decode():
    raise NotImplementedError


def _parse_function(example_proto):
    features = {
        'experiment_length': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string),
        'seg': tf.FixedLenFeature([], tf.string),
        'object_segments': tf.FixedLenFeature([], tf.string),
        'gripperpos': tf.FixedLenFeature([], tf.string),
        'objpos': tf.FixedLenFeature([], tf.string),
        'objvel': tf.FixedLenFeature([], tf.string),
        'experiment_id': tf.FixedLenFeature([], tf.int64),
        'n_total_objects': tf.FixedLenFeature([], tf.int64),
        'n_manipulable_objects': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    experiment_length = parsed_features['experiment_length']
    n_manipulable_objects = parsed_features['n_manipulable_objects']
    experiment_id = parsed_features['experiment_id']
    n_total_objects = parsed_features['n_total_objects']

    img = tf.decode_raw(parsed_features['img'], out_type=tf.uint8)
    img = tf.reshape(img, tf.stack([experiment_length, 120, 160, 3]))

    seg = tf.decode_raw(parsed_features['seg'], out_type=tf.uint8)
    seg = tf.reshape(seg, tf.stack([experiment_length, 120, 160]))

    object_segments = tf.decode_raw(parsed_features['object_segments'], out_type=tf.uint8)
    if seg_only_for_initialization:
        object_segments = tf.reshape(object_segments, tf.stack([n_total_objects, 120, 160, 4]))
    else:
        object_segments = tf.reshape(object_segments, tf.stack([n_total_objects*experiment_length, 120, 160, 4]))

    gripperpos = tf.decode_raw(parsed_features['gripperpos'], out_type=tf.float64)
    gripperpos = tf.reshape(gripperpos, tf.stack([experiment_length, 3]))

    objpos = tf.decode_raw(parsed_features['objpos'], out_type=tf.float64)
    objpos = tf.reshape(objpos, tf.stack([experiment_length*n_manipulable_objects, 3]))

    objvel = tf.decode_raw(parsed_features['objvel'], out_type=tf.float64)
    objvel = tf.reshape(objvel, tf.stack([experiment_length*n_manipulable_objects, 3]))

    return experiment_length, img, seg, object_segments, gripperpos, objpos, objvel, experiment_id, n_total_objects, n_manipulable_objects


if __name__ == '__main__':
    filenames = ["../data/destination/train1_of_8.tfrecords", "../data/destination/train2_of_8.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)