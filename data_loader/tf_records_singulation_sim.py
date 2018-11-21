import tensorflow as tf
import numpy as np
import os
from skimage import img_as_uint
from utils.utils import chunks, get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir
from data_prep.segmentation import get_segments_from_experiment_step
from data_prep.segmentation import get_number_of_segment
from sklearn.model_selection import train_test_split
from utils.config import process_config
from utils.utils import get_args


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

def check_if_skip(experiment):
    check_lst = []
    for step in experiment.values():
        check_lst.append(get_number_of_segment(step["seg"]))

    if len(np.unique(check_lst)) is 1:
        # if number of objects vary and should be discarded, go to next experiment
        return False
    return True


def add_experiment_data_to_lists(experiment, identifier, use_object_seg_data_only_for_init, depth_data_provided):

    stop_object_segments = False

    objects_segments = []
    gripperpos = []
    objpos = []
    objvel = []
    depth = []
    img = []
    seg = []

    for j, trajectory_step in enumerate(experiment.values()):
        image_data = {
            'img': trajectory_step['img'],
            'seg': trajectory_step['seg'],
            'depth': trajectory_step['xyz']
        }
        segments = get_segments_from_experiment_step(image_data, depth_data_provided=depth_data_provided)

        keys = ["{}{}".format(i, identifier) for i in range(segments['n_segments'])]
        temp_list = []
        if not use_object_seg_data_only_for_init or not stop_object_segments:
            for k in segments.keys():
                if k in keys:
                    temp_list.append(segments[k])
            stop_object_segments = True
            objects_segments.append(np.stack(temp_list))
        if depth_data_provided:
            depth.append(img_as_uint(trajectory_step['xyz'])) #todo changed

        seg.append(img_as_uint(trajectory_step['seg']))
        img.append(img_as_uint(trajectory_step['img']))
        #seg.append(trajectory_step['seg'].astype(np.int16))
        #img.append(trajectory_step['img'].astype(np.int16))
        gripperpos.append(trajectory_step['gripperpos'])
        objpos.append(np.stack(list(trajectory_step['objpos'].tolist().values())))
        objvel.append(np.stack(list(trajectory_step['objvel'].tolist().values())))

    if depth_data_provided:
        return objects_segments, gripperpos, objpos, objvel, img, seg, depth
    return objects_segments, gripperpos, objpos, objvel, img, seg



def create_tfrecords_from_dir(config, source_path, dest_path, discard_varying_number_object_experiments=True,
                              n_sequences_per_batch = 10, test_size=0.2):
    """

    :param source_path:
    :param dest_path:
    :param name:
    :return:
    """
    use_object_seg_data_only_for_init = config.use_object_seg_data_only_for_init
    depth_data_provided = config.depth_data_provided
    use_compression = config.use_tfrecord_compression

    file_paths = get_all_experiment_file_paths_from_dir(source_path)
    train_paths, test_paths = train_test_split(file_paths, test_size=test_size)
    filenames_split_train = list(chunks(train_paths, n_sequences_per_batch))
    filenames_split_test = list(chunks(test_paths, n_sequences_per_batch))

    filenames = filenames_split_train + filenames_split_test
    train_ids = ["train"] * len(filenames_split_train)
    test_ids = ["test"] * len(filenames_split_test)
    identifiers = np.concatenate([train_ids, test_ids])

    if use_compression:
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    else:
        options = None

    for i, queue in enumerate(zip(filenames, identifiers)):
        all_batches = queue[0]
        name = queue[1]

        loaded_batch = load_all_experiments_from_dir(all_batches)

        filename = os.path.join(dest_path, name + str(i+1) + '_of_' + str(len(filenames)) + '.tfrecords')
        print('Writing', filename)


        identifier = "_object_full_seg_rgb"
        depth = None
        if depth_data_provided:
            identifier = "_object_full_seg_rgb_depth"

        with tf.python_io.TFRecordWriter(path=filename, options=options) as writer:
            for experiment in loaded_batch.values():

                if not experiment:
                    continue

                if discard_varying_number_object_experiments:
                    skip = check_if_skip(experiment)
                    if skip:
                        continue

                number_of_total_objects = get_number_of_segment(experiment[0]['seg'])
                n_manipulable_objects = number_of_total_objects - 2 # container and gripper subtracted (background is removed)
                experiment_id = int(experiment[0]['experiment_id'])

                # all data objects are transformed s.t. for each data a list consisting of 'experiment_length' ndarrays returned,
                # if data is multi-dimensional (e.g. segments for each object per times-step), ndarrays are stacked along first
                # dimension
                if depth_data_provided:
                    objects_segments, gripperpos, objpos, objvel, img, seg, depth = add_experiment_data_to_lists(experiment, identifier,
                                                            use_object_seg_data_only_for_init=use_object_seg_data_only_for_init,
                                                            depth_data_provided= depth_data_provided)
                    depth = [_bytes_feature(i.tostring()) for i in depth]
                else:
                    objects_segments, gripperpos, objpos, objvel, img, seg = add_experiment_data_to_lists(experiment, identifier,
                                                            use_object_seg_data_only_for_init=use_object_seg_data_only_for_init,
                                                            depth_data_provided= depth_data_provided)



                imgs =[_bytes_feature(i.tostring()) for i in img]
                segs = [_bytes_feature(i.tostring()) for i in seg]
                gripperpositions = [_bytes_feature(i.tostring()) for i in gripperpos]

                # concatenate all object positions/velocities into an ndarray per experiment step
                objpos = [_bytes_feature(i.tostring()) for i in objpos]

                objvel = [_bytes_feature(i.tostring()) for i in objvel]

                objects_segments = [_bytes_feature(i.tostring()) for i in objects_segments]

                feature_list = {
                    'img': tf.train.FeatureList(feature=imgs),
                    'seg' : tf.train.FeatureList(feature=segs),
                    'gripperpos': tf.train.FeatureList(feature=gripperpositions),
                    'objpos': tf.train.FeatureList(feature=objpos),
                    'objvel': tf.train.FeatureList(feature=objvel),
                    'object_segments': tf.train.FeatureList(feature=objects_segments)
                }

                if depth_data_provided:
                    feature_list['depth'] = tf.train.FeatureList(feature=depth)

                feature_lists = tf.train.FeatureLists(feature_list=feature_list)

                example = tf.train.SequenceExample(feature_lists=feature_lists)
                example.context.feature['experiment_length'].int64_list.value.append(len(experiment.keys()))
                example.context.feature['experiment_id'].int64_list.value.append(experiment_id)
                example.context.feature['n_total_objects'].int64_list.value.append(number_of_total_objects)
                example.context.feature['n_manipulable_objects'].int64_list.value.append(n_manipulable_objects)

                writer.write(example.SerializeToString())


if __name__ == '__main__':
    args = get_args()
    config = process_config(args.config)
    create_tfrecords_from_dir(config, "../data/source", "../data/destination", test_size=0.2, n_sequences_per_batch=30)
