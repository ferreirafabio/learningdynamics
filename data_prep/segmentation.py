import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_experiment_image_data_from_dir, get_all_experiment_image_data_from_dir, get_all_experiment_image_data_from_dir


def get_segments_from_experiment_step(images):
    """
    Extracts the given segments and the rgb crops at these positions from a given experiment step.
    Args:
        images: a list containing two ndarrays of dimensions WxHxC and WxH while W=width, H=height, C=channels
    Returns:
        a dictionary with Nx2+2 ndarrays while N is the number of segments. The number of segments is doubled since the segments are
        included as segments in the full image as well as a crop. 2 additional arrays are included for the original and the full
        segmentation image. Except the original and the full segmentation arrays, every array consists of 4 channels (seg, R, B, G).

        Example return keys:
            'full_seg' (3 channel ndarray)
            'full_rgb' (3 channel ndarray)
            '0_object_full_seg_rgb' (4 channel ndarray)
            '0_object_crop_seg_rgb' (4 channel ndarray)
            '1_...' (4 channel ndarray)
    """
    assert type(images) == list, "parameter images is not a list, expecting a list"

    seg = None
    rgb = None

    for element in images:
        if type(element) == np.ndarray:
            if element.ndim == 3:
                rgb = element.astype(np.uint8)
            else:
                seg = element.astype(np.uint8)

    n_segments = get_number_of_segment(seg)
    masks = get_segments_indices(seg)
    seg_rgb_data = {
        "n_segments": n_segments,
        "full_seg": seg,
        "full_rgb": rgb
    }

    for i in range(n_segments):
        # get full images
        full_seg_masked = (seg == masks[i]).astype(np.uint8)
        full_rgb_masked = get_segment_by_mask(rgb, full_seg_masked).astype(np.uint8)

        full_seg_masked_expanded = np.expand_dims(full_seg_masked, axis=2)
        full_seg_rgb_masked = np.concatenate((full_seg_masked_expanded, full_rgb_masked), axis=2)

        # channel 0: seg, channel 1..3: rgb
        seg_rgb_data[str(i) + "_object_" + "full_seg_rgb"] = full_seg_rgb_masked

        # get crops
        crop = crop_by_mask(full_seg_masked).astype(np.uint8)
        rgb_crop = get_segment_by_mask(rgb, mask=full_seg_masked, crop=True).astype(np.uint8)

        crop_seg_masked_expanded = np.expand_dims(crop, axis=2)
        crop_seg_rgb_masked = np.concatenate((crop_seg_masked_expanded, rgb_crop), axis=2)

        # channel 0: seg, channel 1..3: rgb
        seg_rgb_data[str(i) + "_object_" + "crop_seg_rgb" ] = crop_seg_rgb_masked

    # todo: remove
    for v in seg_rgb_data.values():
        if type(v) == np.ndarray:
            plt.imshow(v, cmap="Greys")
            plt.show()

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

    """ all experiments """
    all_experiments_image_data = get_all_experiment_image_data_from_dir(source_path, data_type=["rgb", "seg"])
    c = get_segments_from_all_experiments(all_experiments_image_data)
    print("done")

