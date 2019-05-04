import math
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

X_MIN = 0.344
Y_MIN = -0.256
Z_MIN = -0.149
X_MAX = 0.856
Y_MAX = 0.256
Z_MAX = -0.0307

def peak_signal_to_noise_ratio(true, pred, color_depth=None):
    """Image quality metric based on maximal signal power vs. power of the noise.
    Args:
    true: the ground truth image.
    pred: the predicted image.
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


def compute_iou(true, pred):
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def compute_precision(true, pred):
    precisions = []

    for true_obj, pred_obj in zip(true, pred):
        true = np.array(true_obj).flatten()
        pred = np.array(pred_obj).flatten()
        score = precision_score(y_true=true, y_pred=pred)
        precisions.append(score)

    return np.mean(precisions), precisions.index(min(precisions)), precisions.index(max(precisions))


def compute_recall(true, pred):
    recalls = []

    for true_obj, pred_obj in zip(true, pred):
        true = np.array(true_obj).flatten()
        pred = np.array(pred_obj).flatten()
        score = recall_score(y_true=true, y_pred=pred)
        recalls.append(score)

    return np.mean(recalls), recalls.index(min(recalls)), recalls.index(max(recalls))

def compute_f1(true, pred):
    f1_scores = []

    for true_obj, pred_obj in zip(true, pred):
        true = np.array(true_obj).flatten()
        pred = np.array(pred_obj).flatten()
        score = f1_score(y_true=true, y_pred=pred)
        f1_scores.append(score)
    return np.mean(f1_scores), f1_scores.index(min(f1_scores)), f1_scores.index(max(f1_scores))


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def normalize_df_column(df, column_name):
    """ normalizes an entire pandas column (series) in which each row r contains a 3-dim ndarray s.t. all values of r in range (0,1) """
    x_norm = lambda x: (x - X_MIN) / (X_MAX - X_MIN)
    y_norm = lambda y: (y - Y_MIN) / (Y_MAX - Y_MIN)
    z_norm = lambda z: (z - Z_MIN) / (Z_MAX - Z_MIN)

    def _normalize_row(row):
        return np.asarray([x_norm(row[0]), y_norm(row[1]), z_norm(row[2])])

    return df[column_name].apply(_normalize_row)


def is_square(integer):
    root = math.sqrt(integer)
    if int(root + 0.5) ** 2 == integer:
        return True
    else:
        return False


def check_power(N, k):
    if N == k:
        return True
    try:
        return N == k**int(round(math.log(N, k)))
    except Exception:
        return False


def normalize_list(lst):
    """ normalizes a list of 3-dim ndarrays x s.t. all x's contain values in (0,1) """
    x_norm = lambda x: (x - X_MIN) / (X_MAX - X_MIN)
    y_norm = lambda y: (y - Y_MIN) / (Y_MAX - Y_MIN)
    z_norm = lambda z: (z - Y_MIN) / (Z_MAX - Z_MIN)

    return [np.asarray([x_norm(coords[0]), y_norm(coords[1]), z_norm(coords[2])]) for coords in lst]


def normalize_df(df):
    """ normalizes an entire pandas dataframe in which each cell c contains a 3-dim ndarray s.t. c only contains values in (0,1) """
    x_norm = lambda x: (x - X_MIN) / (X_MAX - X_MIN)
    y_norm = lambda y: (y - Y_MIN) / (Y_MAX - Y_MIN)
    z_norm = lambda z: (z - Y_MIN) / (Z_MAX - Z_MIN)

    def _normalize_column(column):
        for index, row_value in column.items():
            column[index] = np.asarray([x_norm(row_value[0]), y_norm(row_value[1]), z_norm(row_value[2])])
        return column

    return df.apply(_normalize_column, axis=1)