import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

def get_iris():

    iris = load_iris()

    for attr in dir(iris):
        print(attr)

    # DESCR
    # data
    # data_module
    # feature_names
    # filename
    # frame
    # target
    # target_names

    # 대문자는 행렬, 소문자는 벡터
    iris_X = iris.data
    iris_y = iris.target
    feature_names = iris.feature_names
    species = iris.target_names
    n_feature = len(feature_names)
    n_species = len(species)

    return iris_X, iris_y, feature_names, species, n_feature, n_species


def iris_visualization1():
    iris_X, iris_y, feature_names, species, n_feature, n_species = get_iris()

    cls_0 = iris_X[iris_y == 0]
    cls_1 = iris_X[iris_y == 1]
    cls_2 = iris_X[iris_y == 2]
    xticks = np.arange(3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for i, ax in enumerate(axes.flat):
        ax.violinplot([cls_0[:, i], cls_1[:, i], cls_2[:, i]],
                      positions=xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(species)
        ax.set_title(feature_names[i], fontsize=20)
        ax.tick_params(labelsize=20)


def histogram_test():
    np.random.seed(0)
    n_data = 500

    data = np.random.normal(0, 1, (n_data,))

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.tick_params(labelsize=30, length=10, width=3)
    ax.hist(data, rwidth=0.9)


def single_pair(axes, row, col, X, y, cls_dict, features):
    color_list = ['purple', 'green', 'orange']

    # histogram
    if row == col:
        data = X[:, row]
        axes[row, col].hist(data, rwidth=0.9)

    # scatter plot
    else:
        for key, val in cls_dict.items():
            axes[row, col].scatter(val[:, col], val[:, row],
                                   edgecolor=f'tab:{color_list[key]}',
                                   color=color_list[key], alpha=0.5)

    # labels
    if col == 0:
        axes[row, col].set(ylabel=features[row])
        axes[row, col].set_ylabel(features[row], fontsize=20)

    if row == len(features)-1:
        axes[row, col].set(xlabel=features[col])
        axes[row, col].set_xlabel(features[col], fontsize=20)


def iris_pairplot():
    iris_X, iris_y, feature_names, species, n_feature, n_species = get_iris()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

    cls_dict = dict()
    for cls in np.unique(iris_y):
        cls_dict[cls] = iris_X[iris_y == cls]

    for i in range(n_feature):
        for j in range(n_feature):
            single_pair(axes, i, j, iris_X, iris_y, cls_dict, feature_names)


''' CODE REFACTORING FOR THE PAIR PLOT '''


def single_pair2(axes, row, col, X, y, features):
    # histogram
    if row == col:
        data = X[:, row]
        axes[row, col].hist(data, rwidth=0.9)

    # scatter plot
    else:
        axes[row, col].scatter(X[:, col], X[:, row], c=y, alpha=0.5)

    # labels
    if col == 0:
        axes[row, col].set_ylabel(features[row], fontsize=20)
    if row == len(features)-1:
        axes[row, col].set_xlabel(features[col], fontsize=20)


def iris_pairplot_clean():
    # code refactoring
    iris_X, iris_y, feature_names, species, n_feature, n_species = get_iris()

    fig, axes = plt.subplots(nrows=n_feature, ncols=n_feature, figsize=(16, 16))

    for i in range(n_feature):
        for j in range(n_feature):
            single_pair2(axes, i, j, iris_X, iris_y, feature_names)


if __name__ == '__main__':
    # iris_visualization1()
    # histogram_test()
    # iris_pairplot()
    iris_pairplot_clean()
    plt.show()

