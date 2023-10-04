import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


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

print('here')

