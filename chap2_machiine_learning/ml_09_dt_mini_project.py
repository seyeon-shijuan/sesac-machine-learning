import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def dt_1():
    # scikit-learn load_diabetes로 regression decision tree 그리기
    diabetes = load_diabetes()
    data, targets = diabetes.data, diabetes.target
    print("data / target shape")
    print(data.shape, targets.shape) # (442, 10) (442,)
    print("="*30)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.1, random_state=11)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # 예측 및 결과 분석
    preds = model.predict(X_test)

    print("depth: ", model.get_depth()) # depth:  17
    print("num of leaves: ", model.get_n_leaves()) # num of leaves:  391
    accuracy = model.score(X_test, y_test)
    print(f"{accuracy = :.4f}") # accuracy = 0.3225

    r2_score = 1 - (((y_test - preds)**2).sum() / ((y_test - y_test.mean())**2).sum())
    print(f"{r2_score = :.4f}") # r2_score = 0.3225


def dt_2():
    # dataset preparation
    df = pd.read_csv("../data/bike_sharing.csv")
    print(df.columns.to_list())
    # ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    # 'workingday', 'weathersit', 'temp', 'atemp', 'hum',
    # 'windspeed', 'casual', 'registered', 'cnt']

    # selected columns out of original columns
    col_names = ['season', 'mnth', 'hr', 'holiday', 'weekday',
                 'workingday', 'weathersit', 'temp', 'atemp',
                 'windspeed']

    X = df[col_names].to_numpy()
    y = df['cnt'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # train
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # predict
    preds = model.predict(X_test)

    # decision tree summary
    print("depth: ", model.get_depth()) # depth:  32
    print("num of leaves: ", model.get_n_leaves()) # num of leaves:  13351
    accuracy = model.score(X_test, y_test)
    print(f"{accuracy = :.4f}") # accuracy = 0.7174


def entropy(p: list):
    tot = sum(p)
    p = np.array(p).astype(dtype='float64')
    p /= tot
    entropy = -np.sum(p * np.log2(p))
    return entropy


def information_gain(parent, child):
    parent_entropy = entropy(parent)
    l_parent = float(sum(parent))

    partition_entropy = []

    for ele in child:
        l_child = float(sum(ele))
        part_ent = entropy(ele)

        curr_ent = l_child / l_parent * part_ent
        partition_entropy.append(curr_ent)

    final_entropy = sum(partition_entropy)
    ig = parent_entropy - final_entropy

    return ig


def get_ig_idx(X, y, col_names):
    ig_list = list()
    parent_uniques, parent_cnts = np.unique(y, return_counts=True)

    for i in range(X.shape[1]):
        curr = X[:, i]
        uq = np.unique(curr)
        children = list()
        for ele in uq:
            ele_idx = (curr == ele)
            curr_y = y[ele_idx]
            uniq, cnts = np.unique(curr_y, return_counts=True)
            # child = [[6], [1, 3]]
            children.append(cnts)

        e = information_gain(parent=parent_cnts, child=children)
        ig_list.append(e)

    ig_list = np.array(ig_list)
    print("col: ", col_names)
    print("gr: ", ig_list)
    max_idx = np.argmax(ig_list)

    return max_idx


def decision_tree_comparison(X, y, col_names):
    max_idx = get_ig_idx(X=X, y=y, col_names=col_names)
    print(f"h1 node: idx {max_idx} {col_names[max_idx]}")
    # h1 node: idx 0 age


    # # h2-1 node separation
    # # data filtration by age - yes
    # uniques = np.unique(X[:, max_idx])
    # to_remain = (X[:, max_idx] == 0)
    # X1 = X[to_remain]
    # X1 = np.delete(X1, max_idx, axis=1)
    # y1 = y[to_remain]
    # col_names.pop(max_idx)

    return max_idx


def dt_3():
    # dataset preparation
    df = pd.read_csv("../data/register_golf_club.csv", index_col=0)
    print(df.columns.to_list())
    # ['age', 'income', 'married', 'credit_score', 'register_golf_club']
    cols = df.columns.to_list()[:-1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    max_idx = decision_tree_comparison(np.array(X), np.array(y), cols)

    DecisionTreeClassifier()



    print('here')





if __name__ == '__main__':
    # dt_1()
    # dt_2()
    dt_3()