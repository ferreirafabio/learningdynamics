import os
import multiprocessing
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

class DataGenerator:
    def __init__(self, config, sess, train=True):
        """
        :param config: the config as a dict
        :param sess: the TF session
        :param train: the mode. If True, Train DataGenerator object is returned, else a Test object.
        :param old_tfrecords: specifies whether the specified path in the config file points to new or old tfrecords
        """
        self.old_tfrecords = config.old_tfrecords
        self.config = config
        self.use_object_seg_data_only_for_init = self.config.use_object_seg_data_only_for_init
        self.depth_data_provided = config.depth_data_provided
        path = self.config.tfrecords_dir
        train_batch_size = self.config.train_batch_size
        test_batch_size = self.config.test_batch_size
        use_compression = self.config.use_tfrecord_compression
        self.use_object_seg_data_only_for_init = self.config.use_object_seg_data_only_for_init

        if train:
            filenames = gfile.Glob(os.path.join(path, "train*"))
            batch_size = train_batch_size
        else:
            filenames = gfile.Glob(os.path.join(path, "test*"))
            batch_size = test_batch_size

        if use_compression:
            compression_type = 'GZIP'
        else:
            compression_type = None

        self.batch_size = batch_size
        self.dataset = tf.data.TFRecordDataset(filenames, compression_type=compression_type)
        self.dataset = self.dataset.prefetch(self.config.dataset_prefetch_size)
        self.dataset = self.dataset.map(self._parse_function, num_parallel_calls=multiprocessing.cpu_count())
        self.dataset = self.dataset.apply(tf.data.experimental.shuffle_and_repeat(30, self.config.n_epochs))



        # Dataset.batch() works only for tensors that all have the same size
        # given shapes are for: img, seg, depth, gripperpos, objpos, objvel, obj_segs, experiment_length, experiment_id, n_total_objects,
        # n_manipulable_objects
        padded_shapes = {
                        'img': (None, 120, 160, 3),
                        'seg': (None, 120, 160),
                        'gripperpos': (None, 3),
                        'objpos': (None, None, 3),
                        'objvel': (None, None, 3),
                        'object_segments': (None, 120, 160, 4),
                        'experiment_length': (),
                        'experiment_id': (),
                        'n_total_objects': (),
                        'n_manipulable_objects': ()
        }

        if not self.use_object_seg_data_only_for_init:
            if self.depth_data_provided:
                padded_shapes['object_segments'] = (None, None, 120, 160, 7)
            else:
                padded_shapes['object_segments'] = (None, None, 120, 160, 4)
        else:
            if self.depth_data_provided:
                padded_shapes['object_segments'] = (None, 120, 160, 7)
            else:
                padded_shapes['object_segments'] = (None, 120, 160, 4)

        if self.depth_data_provided:
            padded_shapes['depth'] = (None, 120, 160, 3)
        self.dataset = self.dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()
        sess.run(self.iterator.initializer)


    def _parse_function(self, example_proto):
        context_features = {
            'experiment_length': tf.FixedLenFeature([], tf.int64),
            'experiment_id': tf.FixedLenFeature([], tf.int64),
            'n_total_objects': tf.FixedLenFeature([], tf.int64),
            'n_manipulable_objects': tf.FixedLenFeature([], tf.int64)
        }

        sequence_features = {
            'img':  tf.FixedLenSequenceFeature([], dtype=tf.string),
            'seg': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'object_segments': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'gripperpos': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'objpos': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'objvel': tf.FixedLenSequenceFeature([], dtype=tf.string)
        }

        if self.depth_data_provided:
            sequence_features['depth'] = tf.FixedLenSequenceFeature([], dtype=tf.string)

        context, sequence = tf.parse_single_sequence_example(example_proto, context_features=context_features,
                                                             sequence_features=sequence_features)

        experiment_length = context['experiment_length']
        n_manipulable_objects = context['n_manipulable_objects']
        experiment_id = context['experiment_id']
        n_total_objects = context['n_total_objects']

        img = tf.decode_raw(sequence['img'], out_type=tf.uint16)
        img = tf.reshape(img, tf.stack([experiment_length, 120, 160, 3]))

        seg = tf.decode_raw(sequence['seg'], out_type=tf.uint16)
        seg = tf.reshape(seg, tf.stack([experiment_length, 120, 160]))

        if self.depth_data_provided:
            depth = tf.decode_raw(sequence['depth'], out_type=tf.uint16)
            depth = tf.reshape(depth, tf.stack([experiment_length, 120, 160, 3]))

            if not self.use_object_seg_data_only_for_init:
                shape_if_depth_provided = tf.stack([experiment_length, n_total_objects, 120, 160, 7])
            else:
                shape_if_depth_provided = tf.stack([n_total_objects, 120, 160, 7])
        else:
            if not self.use_object_seg_data_only_for_init:
                shape_if_depth_provided = tf.stack([experiment_length, n_total_objects, 120, 160, 4])
            else:
                shape_if_depth_provided = tf.stack([n_total_objects, 120, 160, 4])

        gripperpos = tf.decode_raw(sequence['gripperpos'], out_type=tf.float64)
        gripperpos = tf.reshape(gripperpos, tf.stack([experiment_length, 3]))

        # indices: 0=first object,...,2=third object
        objpos = tf.decode_raw(sequence['objpos'], out_type=tf.float64)
        objpos = tf.reshape(objpos, tf.stack([experiment_length, n_manipulable_objects, 3]))

        # the following yields [240,240,240] to broadcast with shape (experiment_length, n_manipulable_objects, 3)
        # might be unneccessary step in case of normalization due to multiplicative scale invariance
        vel_broadcast_tensor = tf.fill((3,), tf.cast(240.0, tf.float64))
        objvel = tf.identity(objpos, name="objvel") * vel_broadcast_tensor  # frequency used: 1/240 --> velocity: pos/time --> pos/(1/f) --> pos*f

        if self.old_tfrecords:
            # object_segments
            img_type = tf.uint8
            object_segments = tf.decode_raw(sequence['object_segments'], out_type=img_type)
        else:
            # object_segments
            img_type = tf.int16
            object_segments = tf.decode_raw(sequence['object_segments'], out_type=img_type)
        object_segments = tf.reshape(object_segments, shape_if_depth_provided)

        if self.config.normalize_data:
            # cast necessary because _normalize_fixed requires minuend and subtrahend to be of the same type
            img_shape = img.get_shape()[-3:]
            seg_shape = seg.get_shape()[-2:]
            object_seg_shape = object_segments.get_shape()[-3:]
            gripperpos_shape = gripperpos.get_shape()[-1:]
            objpos_shape = objpos.get_shape()[-1:]

            img = _normalize_fixed(img, normed_min=0, normed_max=1, shape=img_shape)
            seg = _normalize_fixed(seg, normed_min=0, normed_max=1, shape=seg_shape)

            if self.depth_data_provided:
                depth_shape = depth.get_shape()[-3:]
                depth = _normalize_fixed(depth, normed_min=0, normed_max=1, shape=depth_shape)

            object_segments = _normalize_fixed(object_segments, normed_min=0, normed_max=1, shape=object_seg_shape)

            gripperpos = _normalize_fixed_pos_vel_data(gripperpos, normed_min=0, normed_max=1, shape=gripperpos_shape)
            objpos = _normalize_fixed_pos_vel_data(objpos, normed_min=0, normed_max=1, shape=objpos_shape)
            objvel = _normalize_fixed_pos_vel_data(objvel, normed_min=0, normed_max=1, shape=objpos_shape, scaling_factor=240.0)


        return_dict = {
            'img': img,
            'seg': seg,
            'gripperpos': gripperpos,
            'objpos': objpos,
            'objvel': objvel,
            'object_segments': object_segments,

            'experiment_length': experiment_length,
            'experiment_id': experiment_id,
            'n_total_objects': n_total_objects,
            'n_manipulable_objects': n_manipulable_objects
        }

        if self.depth_data_provided:
            return_dict['depth'] = depth

        return return_dict

    def get_next_batch(self):
        return self.iterator.get_next()


def _normalize_fixed(x, normed_min, normed_max, shape):
    """ this function uses broadcasting to operate over the unspecified dimensions (e.g. experiment_length),
    shape provides the known dimensions over which this function should operate """

    current_min_tensor = tf.fill(shape, tf.reduce_min(x))
    current_max_tensor = tf.fill(shape, tf.reduce_max(x))

    # using broadcasting, e.g. shape (10,5,120,160,7) subtracted by shape (120,160,7) yields (10,5,120,160,7)
    x_normed = (x - current_min_tensor) / (current_max_tensor - current_min_tensor)

    normed_min_tensor = tf.fill(shape, tf.cast(normed_min, x_normed.dtype))
    normed_max_tensor = tf.fill(shape, tf.cast(normed_max, x_normed.dtype))

    x_normed = x_normed * (normed_max_tensor - normed_min_tensor) + normed_min_tensor

    return x_normed


def _normalize_fixed_pos_vel_data(x_inp, normed_min, normed_max, shape, scaling_factor=1.0):
    """ normalizes position and velocity data to a range (normed_min, normed_max).
    The scaling_factor can be used when position data is to be converted into velocity data """
    x_min = 0.344*scaling_factor
    y_min = -0.256*scaling_factor
    z_min = -0.149*scaling_factor
    x_max = 0.856*scaling_factor
    y_max = 0.256*scaling_factor
    z_max = -0.0307*scaling_factor

    if len(x_inp.get_shape()) == 2:
        x = (x_inp[:, 0] - x_min) / (x_max - x_min)
        y = (x_inp[:, 1] - y_min) / (y_max - y_min)
        z = (x_inp[:, 2] - z_min) / (z_max - z_min)
        x_normed = tf.stack([x, y, z], axis=1)
    elif len(x_inp.get_shape()) == 3:
        x = (x_inp[:, :, 0] - x_min) / (x_max - x_min)
        y = (x_inp[:, :, 1] - y_min) / (y_max - y_min)
        z = (x_inp[:, :, 2] - z_min) / (z_max - z_min)
        x_normed = tf.stack([x, y, z], axis=2)
    else:
        raise ValueError("parsing dataset failed because position samples have unexpected shapes")

    normed_min_tensor = tf.fill(shape, tf.cast(normed_min, x_normed.dtype))
    normed_max_tensor = tf.fill(shape, tf.cast(normed_max, x_normed.dtype))

    x_normed = x_normed * (normed_max_tensor - normed_min_tensor) + normed_min_tensor

    return x_normed
