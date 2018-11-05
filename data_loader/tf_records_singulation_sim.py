import tensorflow as tf
import numpy as np
import os
from utils.utils import chunks, get_all_experiment_file_paths_from_dir, load_all_experiments_from_dir
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
        if not seg_only_for_initialization or not stop_object_segments:
            for k in range(segments['n_segments']):
                objects_segments = add_to_dict(dct=objects_segments, data=segments, i=k, key=str(k) + identifier)
                stop_object_segments = True

        # img, seg, gripperpos and objvel is object-unspecific
        seg.append(trajectory_step['seg'])
        img.append(trajectory_step['img'])
        gripperpos.append(trajectory_step['gripperpos'])
        objpos.append(trajectory_step['objpos'].tolist())
        objvel.append(trajectory_step['objvel'].tolist())

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
                    n_manipulable_objects = number_of_total_objects - 2 # container and gripper subtracted (background is removed)
                    experiment_id = int(experiment[0]['experiment_id'])

                    objects_segments, gripperpos, objpos, objvel, img, seg = add_experiment_data_to_lists(
                                                                experiment, objects_segments, gripperpos, objpos, objvel, img, seg,
                                                                identifier, seg_only_for_initialization=seg_only_for_initialization)

                    # can't store a list of lists as tfrecords, therefore concatenate all object lists (which each contain an entire
                    # trajectory (experiment length) and store trajectory length to reconstruct initial per-object list
                    # final list has length number_objects * experiment_length OR 1(if seg only used for init), object 0 has elements
                    # 0..exp_length etc.
                    objects_segments = [list(objects_segments[i]) for i in objects_segments.keys()]
                    objects_segments = [_bytes_feature(lst.tostring()) for objct in objects_segments for lst in objct]
                    #objects_segments = [lst.tobytes() for objct in objects_segments for lst in objct]

                    # maintain shape of (n, 3) with n=experiment length but convert 3d entries into bytes for serialization
                    # convert back with np.frombuffer
                    gripperpos = [array for array in gripperpos]

                    # objvel and objpos get reshaped into a long list of z = n_manipulable_objects * experiment_length,
                    # e.g. for 3 objects: obj1_t, obj2_t, obj3_t, obj1_t+1, ..., obj3_z
                    # to access 2nd object in timestep 20, index is 20*n_manipulable_objects+1
                    objvel = [objct.tobytes() for trajectory_step_objvel in objvel for objct in trajectory_step_objvel.values()]
                    objpos = [objct.tobytes() for trajectory_step_objpos in objpos for objct in trajectory_step_objpos.values()]

                    #feature['gripperpos'] = _bytes_feature(gripperpos)
                    #feature['objpos'] = _bytes_feature(objpos)
                    #feature['objvel'] = _bytes_feature(objvel)

                    imgs =[_bytes_feature(i.tostring()) for i in img]
                    segs = [_bytes_feature(i.tostring()) for i in seg]
                    gripperpositions = [_bytes_feature(i.tostring()) for i in gripperpos]

                    #obj_dict = {}
                    #for i, obj_feature in enumerate(objects_segments):
                    #    obj_dict['obj' + str(i)] = tf.train.FeatureList(feature=obj_feature)

                    feature_list = {
                        'img': tf.train.FeatureList(feature=imgs),
                        'seg' : tf.train.FeatureList(feature=segs),
                        'gripperpos': tf.train.FeatureList(feature=gripperpositions)

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
                              seg_only_for_initialization=True)

    #filenames = ["../data/destination/train1_of_8.tfrecords", "../data/destination/train2_of_8.tfrecords"]
    #dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.map(_parse_function)