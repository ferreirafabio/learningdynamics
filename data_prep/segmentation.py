import numpy as np
from utils.utils import get_experiment_image_data_from_dir, load_image_data_from_dir
import matplotlib.pyplot as plt
from scipy import ndimage

def extract_segments_and_rgb_from_single_image(full_seg, full_rgb):
    full_seg = full_seg.astype(np.uint8)
    full_rgb = full_rgb.astype(np.uint8)

    n_segments = get_number_of_segment(full_seg)
    masks = get_segments_indices(full_seg)
    seg_rgb_data = {
        "full_seg": full_seg,
        "full_rgb": full_rgb
    }

    for i in range(n_segments):
        full_seg_masked = (full_seg == masks[i])
        seg_rgb_data["full_seg" + "_object_" + str(i)] = full_seg_masked
        crop = get_segment_crop(full_seg_masked, tol=0)
        seg_rgb_data["seg_crop" + "_object_" + str(i)] = crop
        #todo
        seg_rgb_data["rgb_crop" + "_object_" + str(i)] = crop

        # plt.imshow(full_seg_masked, cmap='Greys')
        # plt.show()
        #
        # plt.imshow(get_segment_crop(full_seg_masked, tol=0), cmap='Greys')
        # plt.show()


def get_single_segments_of_images(source_path, data_type):
    image_data = load_image_data_from_dir(source_path=source_path, data_type=data_type)

    print(image_data)

def get_number_of_segment(img):
    """ does not count in background """
    return len(np.unique(img))-1

def get_segments_indices(img):
    """ does not count in background and assumes background has always value 0"""
    indices = np.unique(img)
    return np.delete(indices, 0)

def get_segment_crop(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


if __name__ == '__main__':
    source_path = "../data/source"
    #get_single_segments_of_images(source_path=source_path, data_type=["rgb", "seg"])
    data = get_experiment_image_data_from_dir(source_path="../data/source", experiment_number=5, data_type="seg")
    extract_segments_and_rgb_from_single_image(data[0])

