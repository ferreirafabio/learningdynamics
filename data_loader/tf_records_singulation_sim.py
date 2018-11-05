import tensorflow as tf
import numpy as np
import os
from utils.utils import chunks, get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir, convert_list_of_dicts_to_list_by_concat
from data_prep.segmentation import get_segments_from_experiment_step
from data_prep.segmentation import get_number_of_segment
from sklearn.model_selection import train_test_split



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


def add_experiment_data_to_lists(experiment, objects_segments, gripperpos, objpos, objvel, img, seg, identifier,
                                 seg_only_for_initialization=True):

    stop_object_segments = False

    for j, trajectory_step in enumerate(experiment.values()):
        segments = get_segments_from_experiment_step([trajectory_step['img'], trajectory_step['seg']])

        keys = ["{}{}".format(i, identifier) for i in range(segments['n_segments'])]
        temp_list = []
        if not seg_only_for_initialization or not stop_object_segments:
            for k in segments.keys():
                if k in keys:
                    temp_list.append(segments[k])
            stop_object_segments = True
            objects_segments.append(np.stack(temp_list))

        seg.append(trajectory_step['seg'])
        img.append(trajectory_step['img'])
        gripperpos.append(trajectory_step['gripperpos'])
        objpos.append(np.stack(list(trajectory_step['objpos'].tolist().values())))
        objvel.append(np.stack(list(trajectory_step['objvel'].tolist().values())))

    return objects_segments, gripperpos, objpos, objvel, img, seg



def create_tfrecords_from_dir(source_path, dest_path, discard_varying_number_object_experiments=True,
                              n_sequences_per_batch = 10, test_size=0.2, seg_only_for_initialization=True):
    """

    :param source_path:
    :param dest_path:
    :param name:
    :return:
    """
    file_paths = get_all_experiment_file_paths_from_dir(source_path)
    train_paths, test_paths = train_test_split(file_paths, test_size=test_size)
    filenames_split_train = list(chunks(train_paths, n_sequences_per_batch))
    filenames_split_test = list(chunks(test_paths, n_sequences_per_batch))

    filenames = [filenames_split_train, filenames_split_test]
    train_ids = ["train"] * len(filenames_split_train)
    test_ids = ["test"] * len(filenames_split_test)
    identifiers = [train_ids, test_ids]

    for i, queue in enumerate(zip(filenames, identifiers)):
        all_batches = queue[0]
        name = queue[1][i]
        for j, batch in enumerate(all_batches, 1):
            loaded_batch = load_all_experiments_from_dir(batch)

            filename = os.path.join(dest_path, name + str(j) + '_of_' + str(len(all_batches)) + '.tfrecords')
            print('Writing', filename)

            options = None #tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            feature = {}
            identifier = "_object_full_seg_rgb"

            with tf.python_io.TFRecordWriter(path=filename, options=options) as writer:
                for experiment in loaded_batch.values():

                    if not experiment:
                        continue

                    objects_segments = []
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
                    n_manipulable_objects = number_of_total_objects - 2 # container and gripper subtracted (background is removed)
                    experiment_id = int(experiment[0]['experiment_id'])

                    # all data objects are transformed s.t. for each data a list consisting of 'experiment_length' ndarrays returned,
                    # if data is multi-dimensional (e.g. segments for each object per times-step), ndarrays are stacked along first
                    # dimension
                    objects_segments, gripperpos, objpos, objvel, img, seg = add_experiment_data_to_lists(
                                                                experiment, objects_segments, gripperpos, objpos, objvel, img, seg,
                                                                identifier, seg_only_for_initialization=seg_only_for_initialization)


                    #objects_segments = [list(objects_segments[i]) for i in objects_segments.keys()]
                    #objects_segments = [lst.tobytes() for objct in objects_segments for lst in objct]


                    imgs =[_bytes_feature(i.tostring()) for i in img]
                    segs = [_bytes_feature(i.tostring()) for i in seg]
                    gripperpositions = [_bytes_feature(i.tostring()) for i in gripperpos]

                    # concatenate all object positions/velocities into an ndarray per experiment step
                    #objpos = convert_list_of_dicts_to_list_by_concat(objpos)
                    objpos = [_bytes_feature(i.tostring()) for i in objpos]

                    #objvel = convert_list_of_dicts_to_list_by_concat(objvel)
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
                    # merge both dicts
                    #feature_list = {**feature_list, **obj_dict}

                    feature_lists = tf.train.FeatureLists(feature_list=feature_list)

                    example = tf.train.SequenceExample(feature_lists=feature_lists)
                    example.context.feature['experiment_length'].int64_list.value.append(len(experiment.keys()))
                    example.context.feature['experiment_id'].int64_list.value.append(experiment_id)
                    example.context.feature['n_total_objects'].int64_list.value.append(number_of_total_objects)
                    example.context.feature['n_manipulable_objects'].int64_list.value.append(n_manipulable_objects)

                    writer.write(example.SerializeToString())





if __name__ == '__main__':
    create_tfrecords_from_dir("../data/source", "../data/destination", test_size=0.2, n_sequences_per_batch=10,
                              seg_only_for_initialization=False)

    #filenames = ["../data/destination/train1_of_8.tfrecords", "../data/destination/train2_of_8.tfrecords"]
    #dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.map(_parse_function)