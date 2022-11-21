import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random

import c45


class Tree:

    def __init__(self):
        self.tree = dict()
        self.epsilon = 0.03
        # self.x_train: pd.DataFrame = None
        # self.y_train: pd.DataFrame = None
        # self.x_test: pd.DataFrame = None
        # self.y_test: pd.DataFrame = None

    def data_load(self, data_path: str, args=None):
        if args is None:
            args = {}
        args.setdefault('random_seed_feature_names', 0)
        args.setdefault('random_state_split', 12)
        args.setdefault('test_size', 0.2)

        data = pd.read_csv(data_path, sep=';')
        data.loc[data['GRADE'] < 4, 'GRADE'] = 0
        data.loc[data['GRADE'] > 0, 'GRADE'] = 1
        data = data.rename(columns={'GRADE': 'SUCCESSFUL'})
        random.seed(args['random_seed_feature_names'])
        self.feature_names = random.sample(list(data.columns[1:-1]), int(len(data) ** 0.5))
        self.feature_result_name = 'SUCCESSFUL'
        self.data = data
        # data_x = data[self.feature_names]
        # data_y = data['SUCCESSFUL']
        # self.data_train, self.data_test, \
        # self.x_train, self.x_test, \
        # self.y_train, self.y_test = train_test_split(data,
        #                                              data_x,
        #                                              data_y,
        #                                              test_size=args['test_size'], random_state=args['random_state_split'])

    def test(self):
        # data = self.data
        # print(f"{data.shape[0]}")
        # print(data['3'].unique())
        # data = data[data[self.feature_result_name] == 1]
        # print(data)
        # print(data)
        self.train(self.data, self.feature_names, self.feature_result_name)
        print(self.tree)

    def train(self, data: pd.DataFrame, feature_names: list[str], feature_result_name: str):
        label = data[feature_result_name].mode()[0]

        if len(feature_names) == 0 or data[feature_result_name].nunique() == 1:
            return label

        def get_best_feature() -> tuple[str, float]:
            feature_best = feature_names[0]
            max_gain = -1
            for feature_current in feature_names:
                gain = c45.gain_ratio(data, feature_current, feature_result_name)
                if max_gain < gain:
                    max_gain = gain
                    feature_best = feature_current
            return feature_best, max_gain

        try:
            best_feature, gain = get_best_feature()
        except ZeroDivisionError:
            return label

        T = {}
        sub_T = {}
        for value_of_best_feature in data[best_feature].unique():
            sub_data = data.loc[data[best_feature] == value_of_best_feature]
            sub_feature_names = feature_names.copy()
            sub_feature_names.remove(best_feature)

            sub_T[f"{best_feature}-{value_of_best_feature}-{data.shape[0]}"] = self.train(sub_data, sub_feature_names,
                                                                                          feature_result_name)

        T[f"{best_feature}-{value_of_best_feature}-{data.shape[0]}"] = sub_T

        self.tree = T
        return T

    def predict(self, x, tree=None):
        if x.ndim == 2:
            res = []
            for x_ in x:
                res.append(self.predict(x_))
            return res

        if not tree:
            tree = self.tree

        tree_key = list(tree.keys())[0]

        x_feature = tree_key.split("___")[0]

        try:
            x_index = self.feature_names.index(x_feature)  # 从列表中定位索引
        except ValueError:
            return '?'
        x_tree = tree[tree_key]
        for key in x_tree.keys():
            if key.split("___")[0] == x[x_index]:
                tree_key = key
                x_tree = x_tree[tree_key]

        if type(x_tree) == dict:
            return self.predict(x, x_tree)
        else:
            return x_tree
