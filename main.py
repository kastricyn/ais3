import utils


def main():
    tree = utils.Tree()
    tree.data_load('DATA.csv')
    tree.test()


if __name__ == '__main__':
    main()
