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
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=12)

    def train(self):
        pass

    def test(self):
        pass

class Node:
    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []
