import math
import pandas as pd


# кол-во элементов (строк) : data[col] == value
def freq(data: pd.DataFrame, col: str, value):
    return len(data[(data[col] == value)])


def info(data: pd.DataFrame, res_col):
    s = 0  # sum
    for v in set(data[res_col]):  # по всем уникальным значениям в столбце
        p = freq(data, res_col, v) / len(data[res_col])
        s += p * math.log(p, 2)
    return -s


def infox(data: pd.DataFrame, col, res_col):
    s = 0  # sum
    for v in set(data[col]):  # по всем уникальным значениям в столбце
        data_i = data[(data[col] == v)]
        s += len(data_i) / len(data) * info(data_i, res_col)
    return s


def split_info(data: pd.DataFrame, col):
    return info(data, col)


def gain_ratio(data: pd.DataFrame, x, res_col):
    return (info(data, res_col) - infox(data, x, res_col)) / split_info(data, x)
