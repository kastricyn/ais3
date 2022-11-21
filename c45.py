import math
import pandas as pd


# кол-во элементов (строк) : data[col] == value
def freq(data: pd.DataFrame, feature: str | int, value):
    return len(data[(data[feature] == value)])


def info(data: pd.DataFrame, feature_res: str | int):
    s = 0  # sum
    for v in set(data[feature_res]):  # по всем уникальным значениям в столбце
        p = freq(data, feature_res, v) / len(data[feature_res])
        s += p * math.log(p, 2)
    return -s


def info_feature(data: pd.DataFrame, feature: str | int, feature_res: str | int):
    s = 0  # sum
    for v in set(data[feature]):  # по всем уникальным значениям в столбце
        data_i = data[(data[feature] == v)]
        s += len(data_i) / len(data) * info(data_i, feature_res)
    return s


def split_info(data: pd.DataFrame, feature: str | int):
    return info(data, feature)


def gain_ratio(data: pd.DataFrame, feature: str | int, feature_res: str | int):
    return (info(data, feature_res) - info_feature(data, feature, feature_res)) / split_info(data, feature)
