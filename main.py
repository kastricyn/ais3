import random

import pandas as pd
from sklearn.model_selection import train_test_split

import utils


def main():
    tree = utils.Tree()
    data_train, data_test, feature_names, feature_result_name = data_prepare('DATA.csv')
    tree.train(data_train, feature_names, feature_result_name)
    predict = tree.predict(data_test)
    # print(predict, data_test[feature_result_name])
    utils.Metrics.print_metrics(data_test[feature_result_name], predict)
    utils.Metrics.draw_plt(data_test[feature_result_name], predict)


def data_prepare(data_path: str, args: dict = None):
    if args is None:
        args = {}
    args.setdefault('random_seed_feature_names', None)
    args.setdefault('random_state_split', None)
    args.setdefault('test_size', 0.2)

    data = pd.read_csv(data_path, sep=';')
    # data['GRADE'] = data['GRADE'].astype(int)
    data.loc[data['GRADE'] < 4, 'GRADE'] = 0
    data.loc[data['GRADE'] > 0, 'GRADE'] = 1
    data = data.rename(columns={'GRADE': 'SUCCESSFUL'})
    random.seed(args['random_seed_feature_names'])
    feature_names = random.sample(list(data.columns[1:-1]), int(len(data) ** 0.5))
    feature_result_name = 'SUCCESSFUL'

    data_train, data_test = train_test_split(data,
                                                 test_size=args['test_size'], random_state=args['random_state_split'])

    return data_train, data_test, feature_names, feature_result_name


if __name__ == '__main__':
    main()
