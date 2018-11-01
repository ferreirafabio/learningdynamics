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
    raise NotImplementedError


def get_single_segments_of_images(seg_images):
    raise NotImplementedError

def get_number_of_segment(img):
    """ no background d """
    return np.unique(img)


if __name__ == '__main__':
    data = get_experiment_image_data_from_dir(source_path="../data/source", experiment_number=5, type="seg")
    img = data[0]
    img = img > img.mean()
    print()

