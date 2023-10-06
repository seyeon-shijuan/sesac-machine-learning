import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import seaborn as sns

plt.style.use('seaborn')

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


def box_plot_test():
    n_student = 100
    math_scores = np.random.normal(loc=50, scale=10, size=(100,))

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    # axes = axes.flatten()

    ''' basics '''
    axes[0, 0].set_title('basic')
    axes[0, 0].boxplot(math_scores)
    axes[0, 1].set_title('notch')
    axes[0, 1].boxplot(math_scores, notch=True)
    axes[0, 2].set_title('whisker')
    axes[0, 2].boxplot(math_scores, notch=True, whis=2)
    axes[0, 3].set_title('sym')
    axes[0, 3].boxplot(math_scores, notch=True, whis=1, sym='bx')

    '''showfliers, vert, median props'''
    axes[1, 0].set_title('showfliers')
    axes[1, 0].boxplot(math_scores, notch=True, showfliers=False)
    axes[1, 1].set_title('vert')
    axes[1, 1].boxplot(math_scores, notch=True, showfliers=False, vert=False)

    median_props = {'linewidth': 2, 'color': 'k'}
    axes[1, 2].set_title('median_props')
    axes[1, 2].boxplot(math_scores, medianprops=median_props)

    box_props = {'linestyle': '--', 'color': 'k', 'alpha': 0.7}
    axes[1, 3].set_title("box props")
    axes[1, 3].boxplot(math_scores, medianprops=median_props, boxprops=box_props)

    whisker_props = {'linestyle': '--', 'color': 'tab:blue', 'alpha': 0.8}
    axes[2, 0].set_title("whiskerprops, capprops")
    axes[2, 0].boxplot(math_scores, medianprops=median_props, boxprops=box_props,
                       whiskerprops=whisker_props, capprops=whisker_props)


    ''' hstacked data '''
    n_student = 100
    math_scores = np.random.normal(loc=50, scale=15, size=(100, 1))
    chem_scores = np.random.normal(loc=70, scale=10, size=(n_student, 1))
    phy_scores = np.random.normal(loc=30, scale=12, size=(n_student, 1))
    pro_scores = np.random.normal(loc=80, scale=5, size=(n_student, 1))

    data = np.hstack((math_scores, chem_scores, phy_scores, pro_scores))
    axes[2, 1].set_ylim([0, 100])
    axes[2, 1].set_title("hstacked data")
    axes[2, 1].boxplot(data)

    ''' major minor tick '''
    axes[2, 2].set_title("major minor tick")
    # n_student = 100
    math_scores = np.random.normal(loc=50, scale=15, size=(100, 1))
    english_scores = np.random.normal(loc=70, scale=10, size=(100, 1))
    physics_scores = np.random.normal(loc=30, scale=12, size=(100, 1))
    programming_scores = np.random.normal(loc=80, scale=5, size=(100, 1))
    data = np.hstack((math_scores, english_scores, physics_scores, programming_scores))
    median_props2 = {'linewidth': 1, 'color': 'tab:red'}
    box_props2 = {'linewidth': 1.5, 'color': 'k', 'alpha': 0.7}
    whisker_props2 = {'linestyle': '--', 'color': 'tab:blue'}

    # labels
    labels = ['Math', 'English', 'Physics', 'Programming']
    axes[2, 2].set_xticklabels(labels)
    # axes[2, 2].tick_params(labelsize=20)
    axes[2, 2].tick_params(axis='x', rotation=10)

    axes[2, 2].boxplot(data, labels=labels,
                       notch=True, medianprops=median_props2, boxprops=box_props2,
                       whiskerprops=whisker_props2, capprops=whisker_props2)

    axes[2, 2].set_ylim([0, 100])

    major_yticks = np.arange(0, 101, 20)
    minor_yticks = np.arange(0, 101, 5)

    axes[2, 2].set_yticks(major_yticks)
    axes[2, 2].set_yticks(minor_yticks, minor=True)

    axes[2, 2].grid(axis='y', linewidth=2)
    axes[2, 2].grid(axis='y', which='minor', linewidth=2, linestyle=':')
    axes[2, 2].grid(axis='x', linewidth=0)

    ''' violin plot '''
    n_group = np.arange(len(labels))
    axes[2, 3].set_title("violin plot")
    axes[2, 3].violinplot(data, positions=n_group)
    axes[2, 3].set_xticks(n_group)
    axes[2, 3].set_xticklabels(labels)
    axes[2, 3].tick_params(axis='x', rotation=10)

    # axes[2, 2].tick_params(labelsize=20, bottom=False, labelbottom=False) # 라벨 지우기
    # fig.subplots_adjust(hspace=0.1) # horizontal space(위아래)를 10%로 줄이는 것
    fig.tight_layout()


if __name__ == '__main__':
    # iris_visualization1()
    # histogram_test()
    # iris_pairplot()
    # iris_pairplot_clean()
    box_plot_test()
    plt.show()

