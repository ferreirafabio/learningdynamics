import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import gfile


class DataGenerator:
    def __init__(self, config, train=True):
        self.config = config
        self.seg_only_for_initialization = self.config["seg_only_for_initialization"]
        path = self.config["tfrecords_dir"]

        if train:
            filenames = gfile.Glob(os.path.join(path, "train*"))
        else:
            filenames = gfile.Glob(os.path.join(path, "valid*"))

        self.dataset = tf.data.TFRecordDataset(filenames)
        self.dataset = self.dataset.map(self._parse_function)
        # Dataset.batch() works only for tensors that all have the same size
        # given shapes are for: img, seg, gripperpos, objpos, objvel, obj_segs, experiment_length, experiment_id, n_total_objects,
        # n_manipulable_objects
        #self.dataset = self.dataset.padded_batch(1, padded_shapes=( (None, 120, 160, 3), (None, 120, 160), (None, 3), (None, None, 3),
        #                                        (None, None, 3), (None, None, 120, 160, 4), (), (), (), () ))
        self.dataset = self.dataset.padded_batch(1, padded_shapes=( (None, None, 120, 160, 4) ))

        self.iterator = self.dataset.make_initializable_iterator()

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]


    def _parse_function(self, example_proto):
        context_features = {
            'experiment_length': tf.FixedLenFeature([], tf.int64),
            'experiment_id': tf.FixedLenFeature([], tf.int64),
            'n_total_objects': tf.FixedLenFeature([], tf.int64),
            'n_manipulable_objects': tf.FixedLenFeature([], tf.int64)
        }

        sequence_features = {
            "img":  tf.FixedLenSequenceFeature([], dtype=tf.string),
            'seg': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'object_segments': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'gripperpos': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'objpos': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'objvel': tf.FixedLenSequenceFeature([], dtype=tf.string)
        }

        context, sequence = tf.parse_single_sequence_example(example_proto, context_features=context_features,
                                                             sequence_features=sequence_features)

        experiment_length = context['experiment_length']
        n_manipulable_objects = context['n_manipulable_objects']
        experiment_id = context['experiment_id']
        n_total_objects = context['n_total_objects']

        img = tf.decode_raw(sequence['img'], out_type=tf.uint8)
        img = tf.reshape(img, tf.stack([experiment_length, 120, 160, 3]))

        seg = tf.decode_raw(sequence['seg'], out_type=tf.uint8)
        seg = tf.reshape(seg, tf.stack([experiment_length, 120, 160]))

        gripperpos = tf.decode_raw(sequence['gripperpos'], out_type=tf.float64)
        gripperpos = tf.reshape(gripperpos, tf.stack([experiment_length, 3]))

        # indices: 0=first object,...,2=third object
        objpos = tf.decode_raw(sequence['objpos'], out_type=tf.float64)
        objpos = tf.reshape(objpos, tf.stack([experiment_length, n_manipulable_objects, 3]))

        # indices: 0=first object,...,2=third object
        objvel = tf.decode_raw(sequence['objvel'], out_type=tf.float64)
        objvel = tf.reshape(objvel, tf.stack([experiment_length, n_manipulable_objects, 3]))


        object_segments = tf.decode_raw(sequence['object_segments'], out_type=tf.uint8)
        if self.seg_only_for_initialization:
            object_segments = tf.reshape(object_segments, tf.stack([1, n_total_objects, 120, 160, 4]))
        else:
            object_segments = tf.reshape(object_segments, tf.stack([experiment_length, n_total_objects, 120, 160, 4]))

        #return img, seg, gripperpos, objpos, objvel, object_segments, experiment_length, experiment_id, n_total_objects,\
        #       n_manipulable_objects

        return object_segments