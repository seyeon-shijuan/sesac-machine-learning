from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn import tree


def iris_test():
    iris = load_iris()
    data, targets = iris.data, iris.target
    print("data / target shape")
    print(data.shape, targets.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=11)
    print(f"{type(X_train) = } / {X_train.shape = }")
    print(f"{type(X_test) = } / {X_test.shape = }")
    print(f"{type(y_train) = } / {y_train.shape = }")
    print(f"{type(y_test) = } / {y_test.shape = }")

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print("depth: ", model.get_depth())
    print("num of leaves: ", model.get_n_leaves())

    print("--"*30)

    accuracy = model.score(X_test, y_test)
    print(f"{accuracy = :.4f}")

    # for attr in dir(model):
    #     if not attr.startswith("_"):
    #         print(attr)

    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, class_names=iris.target_names,
                   feature_names=iris.feature_names,
                   impurity=True,
                   filled=True,
                   rounded=True)

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    iris_test()