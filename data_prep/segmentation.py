import numpy as np
import matplotlib.pyplot as plt
from utils.io import get_all_experiment_image_data_from_dir, get_experiment_image_data_from_dir, save_image_data_to_disk
from utils.utils import get_all_experiment_image_data_from_dir
from skimage import img_as_ubyte

def get_segments_from_experiment_step(images, depth_data_provided=False):
    """
    Extracts the given segments and the rgb crops at these positions from a given experiment step. If depth image data is provided,
    these extractions are also applied for the depth images. In this case, the full_seg_rgb and i_object_full_seg_rgb get 3
    additional channels.
    Args:
        images: a dict containing ndarrays of dimensions WxHxC or WxH while W=width, H=height, C=channels
    Returns:
        a dictionary with Nx2+2 ndarrays while N is the number of segments. The number of segments is doubled since the segments are
        included as segments in the full image as well as a crop. 2 additional arrays are included for the original and the full
        segmentation image. If depth data is provided, 3 additional arrays are included. The original and the full segmentation
        arrays consist of 4 channels (seg, R, B, G) or 7 channels (if depth data provided)

        Example return keys:
            'full_seg' (1 channel ndarray)
            'full_rgb' (3 channel ndarray)
            'full_depth' (3 channel ndarray)
            '0_object_full_seg_rgb' (4 channel ndarray) (if depth: 7 channel)
            '0_object_crop_seg_rgb' (4 channel ndarray) see above
            '1_...' (4 channel ndarray) see above
    """
    assert type(images) == dict, "parameter images is not a dict, expecting a dict"

    seg = images['seg']
    rgb = images['img']
    depth = None
    full_depth_masked = None

    n_segments = get_number_of_segment(seg)
    masks = get_segments_indices(seg)
    seg_rgb_data = {
        "n_segments": n_segments,
        "full_seg": images['seg'],
        "full_rgb": images['img']
    }

    if depth_data_provided:
        assert 'depth' in images.keys(), "no depth data given but flag 'depth_data_provided' is set to True"
        seg_rgb_data['full_depth'] = images['depth']
        depth = images['depth']
        if depth.dtype == np.float32:
            depth = img_as_ubyte(images['depth']) # depth image requires int16 rescaling

    for i in range(n_segments):
        # get full images
        full_seg_masked = (seg == masks[i]).astype(np.uint8)
        full_rgb_masked = img_as_ubyte(get_segment_by_mask(rgb, full_seg_masked))

        if depth_data_provided:
            full_depth_masked = img_as_ubyte(get_segment_by_mask(depth, full_seg_masked))


        full_seg_masked_expanded = np.expand_dims(full_seg_masked, axis=2)
        identifier = "full_seg_rgb"
        if depth_data_provided:
            full_seg_rgb_depth_masked = np.concatenate((full_seg_masked_expanded, full_rgb_masked, full_depth_masked), axis=2)
            identifier = "full_seg_rgb_depth"
        else:
            full_seg_rgb_depth_masked = np.concatenate((full_seg_masked_expanded, full_rgb_masked), axis=2)

        # channel 0: seg, channel 1..3: rgb
        seg_rgb_data[str(i) + "_object_" + identifier] = full_seg_rgb_depth_masked

        """ get crops """
        crop = img_as_ubyte(crop_by_mask(full_seg_masked))
        rgb_crop = img_as_ubyte(get_segment_by_mask(rgb, mask=full_seg_masked, crop=True))

        depth_crop = None
        identifier = "crop_seg_rgb"
        if depth_data_provided:
            depth_crop = img_as_ubyte(get_segment_by_mask(depth, mask=full_seg_masked, crop=True))

        crop_seg_masked_expanded = np.expand_dims(crop, axis=2)
        if depth_data_provided:
            crop_rgb_seg_depth_masked = np.concatenate((rgb_crop, crop_seg_masked_expanded, depth_crop), axis=2)
            identifier = "crop_seg_rgb_depth"
        else:
            crop_rgb_seg_depth_masked = np.concatenate((rgb_crop, crop_seg_masked_expanded), axis=2)

        # channel 0: seg, channel 1..3: rgb
        seg_rgb_data[str(i) + "_object_" + identifier] = crop_rgb_seg_depth_masked

    # # todo: remove
    # for v in seg_rgb_data.values():
    #    if type(v) == np.ndarray:
    #        plt.imshow(v)
    #        plt.show()

    return seg_rgb_data


def get_segments_from_single_experiment(data):
    segment_data_experiment = []
    for lst in data:
        segment_data_experiment.append(get_segments_from_experiment_step(lst))
    return segment_data_experiment

def get_segments_from_all_experiments(data):
    img_data = []
    for lst in data:
        img_data.append(get_segments_from_single_experiment(lst))
    return img_data


def get_number_of_segment(img):
    """ does not count in background """
    return len(np.unique(img))-1


def get_segments_indices(img):
    """ does not count in background and assumes background has always value 0"""
    indices = np.unique(img)
    return np.delete(indices, 0)


def get_segment_by_mask(img, mask, crop=False):
    new_img = img.copy()
    new_img[mask == False] = 0
    if crop:
        new_img = crop_by_mask(new_img, mask)
    return new_img


def crop_by_mask(img, mask=None):
    new_img = img.copy()
    if mask is None:
        mask = new_img
    return new_img[np.ix_(mask.any(1), mask.any(0))]



if __name__ == '__main__':
    source_path = "../data/source"
    single_experiment_image_data = get_experiment_image_data_from_dir(source_path="../data/source", experiment_id=60, data_type=["seg",
                                                                                                                               "rgb"])
    """ assumes the assignment order of segments to objects is the same within an experiment """
    """ single step """
    a = get_segments_from_experiment_step(single_experiment_image_data[0])
    """ single experiment """
    b = get_segments_from_single_experiment(single_experiment_image_data)

    save_image_data_to_disk(image_data=a, destination_path='../data/', store_gif=True)
    """ all experiments """
    all_experiments_image_data = get_all_experiment_image_data_from_dir(source_path, data_type=["rgb", "seg"])
    c = get_segments_from_all_experiments(all_experiments_image_data)
    print("done")

