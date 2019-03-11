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


    def _parse_function(self, example_proto, normalize=True):
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

        # indices: 0=first object,...,2=third object
        #objvel = tf.decode_raw(sequence['objvel'], out_type=tf.float64)
        #objvel = tf.reshape(objvel, tf.stack([experiment_length, n_manipulable_objects, 3]))


        if self.old_tfrecords:
            img_type = tf.uint8
            fixed_range = [np.iinfo(np.uint8).min, np.iinfo(np.uint8).max]
            object_segments = tf.decode_raw(sequence['object_segments'], out_type=img_type)
        else:
            img_type = tf.int16
            fixed_range = [np.iinfo(np.int16).min, np.iinfo(np.int16).max]
            object_segments = tf.decode_raw(sequence['object_segments'], out_type=img_type)
        object_segments = tf.reshape(object_segments, shape_if_depth_provided)

        if normalize:
            mult = tf.stack([experiment_length])
            fixed_range_tensor = tf.reshape(tf.tile(fixed_range, mult), [mult[0], tf.shape(fixed_range)[0]])
            #print(fixed_range_tensor.get_shape())
            img = _normalize_fixed(img, current_range=fixed_range_tensor, normed_range=[[0, 1]])

        return_dict = {
            'img': img,
            'seg': seg,
            'gripperpos': gripperpos,
            'objpos': objpos,
            'objvel': tf.identity(objpos, name="objvel") * 240,  # frequency used: 1/240 --> velocity: pos/time --> pos/(1/f) --> pos*f
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


def _normalize_fixed(x, current_range, normed_range):
    normed_range = np.asarray(normed_range)
    # subtract over first dimension
    shape = [1] * len(x.get_shape())
    shape[0] = -1

    """ generates the min and max tensors of shape (?,) while "?" typically being e.g. the experiment length """
    current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    current_min, current_max = tf.cast(current_min, x.dtype), tf.cast(current_max, x.dtype)

    """ reshape from (?,1) to (?, (-1, n))) while n = number of dimensions of x minus 1 """
    current_min, current_max = tf.reshape(current_min, shape), tf.reshape(current_max, shape)
    normed_min, normed_max = tf.reshape(normed_min, shape), tf.reshape(normed_max, shape)

    x_normed = (x - current_min) / (current_max - current_min)
    normed_min, normed_max = tf.cast(normed_min, x_normed.dtype), tf.cast(normed_max, x_normed.dtype)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed