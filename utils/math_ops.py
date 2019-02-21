import math

import numpy as np


def normalize_df_column(df, column_name):
    """ normalizes an entire pandas column (series) in which each row r contains a 3-dim ndarray s.t. all values of r in range (0,1) """
    x_min = 0.344
    y_min = -0.256
    z_min = -0.149
    x_max = 0.856
    y_max = 0.256
    z_max = -0.0307

    x_norm = lambda x: (x - x_min) / (x_max - x_min)
    y_norm = lambda y: (y - y_min) / (y_max - y_min)
    z_norm = lambda z: (z - z_min) / (z_max - z_min)

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
    x_min = 0.344
    y_min = -0.256
    z_min = -0.149
    x_max = 0.856
    y_max = 0.256
    z_max = -0.0307

    x_norm = lambda x: (x - x_min) / (x_max - x_min)
    y_norm = lambda y: (y - y_min) / (y_max - y_min)
    z_norm = lambda z: (z - z_min) / (z_max - z_min)

    return [np.asarray([x_norm(coords[0]), y_norm(coords[1]), z_norm(coords[2])]) for coords in lst]


def normalize_df(df):
    """ normalizes an entire pandas dataframe in which each cell c contains a 3-dim ndarray s.t. c only contains values in (0,1) """
    x_min = 0.344
    y_min = -0.256
    z_min = -0.149
    x_max = 0.856
    y_max = 0.256
    z_max = -0.0307

    x_norm = lambda x: (x - x_min) / (x_max - x_min)
    y_norm = lambda y: (y - y_min) / (y_max - y_min)
    z_norm = lambda z: (z - z_min) / (z_max - z_min)

    def _normalize_column(column):
        for index, row_value in column.items():
            column[index] = np.asarray([x_norm(row_value[0]), y_norm(row_value[1]), z_norm(row_value[2])])
        return column

    return df.apply(_normalize_column, axis=1)