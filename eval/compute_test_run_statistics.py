import numpy as np

def compute_psnr(image_dicts, n_objects):
    dict_names = ['predicted_rgb', 'predicted_seg', 'predicted_seg', 'target_rgb', 'target_seg', 'target_seg']
    object_names = ['object_' + str(i) for i in range(n_objects)]

    for k, v in results[0][0].items():
        pair_dict = {}
        for obj_name in object_names:
            if obj_name in k and ('predicted' in k or 'target' in k):
                pair_dict[k] = v
        pred_target_pairs.append(pair_dict)


def peak_signal_to_noise_ratio(true, pred, color_depth=255):
    """Image quality metric based on maximal signal power vs. power of the noise.
    Args:
    true: the ground truth image.
    pred: the predicted image.
    color_depth: the color depth, defaults to 255
    Returns:
    peak signal to noise ratio (PSNR)
    """
    assert color_depth is not None, "please specify color depth"

    mse = mean_squared_error(true, pred)
    if mse == 0 or mse is None:
        psnr = float('inf')
    else:
        psnr = 10.0 * np.log(np.square(color_depth) / mse)

    return psnr


def mean_squared_error(y, y_pred):
    return np.mean((y.flatten() - y_pred.flatten()) ** 2)