import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random


class Tree:
    def __init__(self, data_path: str):
        data = pd.read_csv(data_path, sep=';')
        data[data['GRADE'] < 4] = 0
        data[data['GRADE'] > 0] = 1
        data = data.rename(columns={'GRADE': 'SUCCESSFUL'})
        columns = random.sample(list(data.columns[1:-1]), int(len(data) ** 0.5))
        data_x = data[columns]
        data_y = data['SUCCESSFUL']
        self.x_train, x_test, self.y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=12)
        self.tree = {}

    def train(self):
        x = self.x_train
        y = self.y_train

        if len(set(y)) == 1:
            return y[0]

        label_1, label_2 = set(y)
        max_label = label_1 if np.sum(y == label_1) > np.sum(y == label_2) else label_2
        # print("max_label", max_label)

        if len(x[0]) == 0:
            return max_label
        if depth > self.depth:
            return max_label

        if len(y) < self.min_leaf_size:
            return max_label

        best_feature_index = 0
        max_gain = 0
        for feature_index in range(len(X[0])):
            gain = self.information_gain(X[:, feature_index], y)
            if max_gain < gain:
                max_gain = gain
                best_feature_index = feature_index

        # print(max_gain)

        if max_gain < self.eps:
            return max_label

        T = {}
        sub_T = {}
        for best_feature in set(X[:, best_feature_index]):
            '''
            best_feature：某个特征下的特征类别
            '''
            sub_y = y[X[:, best_feature_index] == best_feature]
            sub_X = X[X[:, best_feature_index] == best_feature]
            sub_X = np.delete(sub_X, best_feature_index, 1)  # 删除最佳特征列

            sub_T[best_feature + "___" + str(len(sub_X))] = self._built_tree(sub_X, sub_y, depth + 1)  # 关键代码

        T[self.feature_names[best_feature_index] + "___" + str(len(X))] = sub_T  # 关键代码

        self.tree = T

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

    def test(self):
        pass

