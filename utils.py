from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, RocCurveDisplay, \
    precision_recall_curve, PrecisionRecallDisplay, auc, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd

import c45


class Tree:

    def __init__(self):
        self.tree = dict()
        self.epsilon = 0.03

    def test(self):
        # print(self.data)
        self.train(self.data, self.feature_names, self.feature_result_name)
        predict = self.predict(self.data)
        print(predict)
        self.data['predict_class'] = predict
        print(self.data)

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
        for value_of_best_feature in data[best_feature].unique():
            sub_data = data.loc[data[best_feature] == value_of_best_feature]
            sub_feature_names = feature_names.copy()
            sub_feature_names.remove(best_feature)

            T[f"{best_feature}-{value_of_best_feature}-{data.shape[0]}"] = self.train(sub_data, sub_feature_names,
                                                                                      feature_result_name)

        self.tree = T
        return T

    def predict(self, data: pd.DataFrame, tree=None) -> list:
        if tree is None:
            tree = self.tree
        result = []
        data = data.reset_index()
        for index, row in data.iterrows():
            def getResultForRow(tree=None):
                feature = list(tree.keys())[0].split("-")[0]
                for key in tree:
                    if str(key).startswith(f"{feature}-{row[feature]}"):
                        if isinstance(tree[key], dict):
                            getResultForRow(tree[key])
                        else:
                            result.append(tree[key])
                        break
                else:
                    result.append(-1)

            getResultForRow(tree)
        return result


class Metrics:
    @classmethod
    def print_metrics(cls, data_predicted: list, data_expect: list):
        print("accuracy_score:", accuracy_score(data_expect, data_predicted))
        print("precision_score:", precision_score(data_expect, data_predicted, average='macro', zero_division=0))
        print("recall_score:", recall_score(data_expect, data_predicted, average='macro'))

    @classmethod
    def draw_plt(cls, data_predict, data_expect):
        fpr, tpr, _ = roc_curve(data_predict, data_expect)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        precision, recall, _ = precision_recall_curve(data_predict, data_expect)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        auc_roc = auc(fpr, tpr)
        auc_pr = average_precision_score(data_predict, data_expect)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        plt.show()
