import numpy as np


def convert_batch_to_list(batch, fltr):
    """
    Args:
        batch:
        fltr: a list of words specifying the keys to keep in the list
    Returns:

    """
    assert type(batch) == dict
    data = []
    for batch_element in batch.values():
        sublist = []
        for i in batch_element.values():
            sublist.append([v for k, v in i.items() for i in fltr if i in k])
        data.append(sublist)
    return data


def convert_dict_to_list(dct):
    """ assumes a dict of subdicts of which each subdict only contains one key containing the desired data """
    lst = []
    for value in dct.values():
        if len(value) > 1:
            lst.append(list(value.values()))
        else:
            element = next(iter(value.values())) # get the first element, assuming the dicts contain only the desired data
            lst.append(element)
    return lst


def convert_dict_to_list_subdicts(dct, length):
    list_of_subdicts = []
    for i in range(length):
        batch_item_dict = {}
        for k, v in dct.items():
            batch_item_dict[k] = v[i]
        list_of_subdicts.append(batch_item_dict)
    return list_of_subdicts


def convert_list_of_dicts_to_list_by_concat(lst):
    """ concatenate all entries of the dicts into an ndarray and append them into a total list """
    total_list = []
    for dct in lst:
        sub_list = []
        for v in list(dct.values()):
            sub_list.append(v)
        sub_list = np.concatenate(sub_list)
        total_list.append(sub_list)
    return total_list


def convert_float_image_to_int16_legacy(float_image): #todo: remove wrong (65k vs 255) conversion when creating new tfrecords
    dt = float_image.dtype
    float_image = float_image.astype(dt) / float_image.max()
    float_image = 255 * float_image
    return float_image.astype(np.int16)