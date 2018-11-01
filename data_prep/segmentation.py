import tensorflow as tf
import numpy as np
import os
import json
import math
import pathlib
from tensorflow.python.platform import gfile
from utils.utils import get_experiment_image_data_from_dir
import matplotlib.pyplot as plt
from scipy import ndimage

def get_single_segments_of_image(seg_image):
    img = seg_image.astype(np.uint8)
    n_segments = get_number_of_segment(img)
    masks = get_segments_indices(img)
    masked_images = []
    crops = []
    for i in range(n_segments):
        new_img = (img == masks[i])
        masked_images.append(new_img)
        plt.imshow(new_img, cmap='Greys')
        plt.show()

        crop = get_segment_crop(new_img, tol=0)
        crops.append(crop)
        plt.imshow(get_segment_crop(new_img, tol=0), cmap='Greys')
        plt.show()

    raise NotImplementedError


def get_single_segments_of_images(seg_images):
    raise NotImplementedError

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
    data = get_experiment_image_data_from_dir(source_path="../data/source", experiment_number=5, type="seg")
    get_single_segments_of_image(data[0])

