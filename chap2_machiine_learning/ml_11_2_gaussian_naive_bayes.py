import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris_df = pd.read_csv('../data/Iris.csv')


def cal_likelihood(x, mean, std):
    likelihood = (1 / (std*np.sqrt(2*np.pi))) * np.exp(-((x - mean) **2) / (2 *(std **2)))
    return likelihood


classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

iris_df_mean = []
iris_df_std = []

# likelihood = list()


test_data = iris_df.iloc[0, :]

fig, axes = plt.subplots(nrows=4, figsize=(8, 12))

for c_idx in range(len(classes)):
    for f_idx in range(len(features)):

        X_cls = iris_df[c_idx*50: 50 *(c_idx+1)]
        X_cls_col = iris_df[c_idx*50: 50 *(c_idx+1)][features[f_idx]]
        iris_mean = iris_df[c_idx*50: 50 *(c_idx+1)][features[f_idx]].mean()
        iris_std = iris_df[c_idx*50: 50 *(c_idx+1)][features[f_idx]].std()
        iris_df_mean.append(iris_mean)
        iris_df_std.append(iris_std)

        # 클래스의 feature 기준 min max로 grid 그리기
        x = np.linspace(iris_df[features[f_idx]].min(), iris_df[features[f_idx]].max(), 1000)

        # 현재 피처의 likelihood list를 구하기 (클래스 x 피처 (3 x 4) 순서로 순회)
        likelihood = cal_likelihood(x, iris_mean, iris_std)

        # line chart plotting
        axes[f_idx].plot(x, likelihood)
        axes[f_idx].set_title(features[f_idx], weight='bold', fontsize=10)

        # test data plotting
        # print(classes[c_i/dx])
        # print(likelihood)
        # axes[f_idx].scatter(test_data[f_idx], likelihood, label=f"{classes[c_idx]} = {likelihood:.3f}")
        axes[f_idx].legend()


plt.tight_layout()
plt.show()


# 이거는 따로돌아가는거라서 iris_mean을 별도의 list로 만들어서 추가해서 돌리지 않는 한
# 기존의 for문안에서 바로바로 line graph를 그리는게 나을 것같음..
# for f_idx in range(len(features)):
#     x = np.linspace(iris_df[features[f_idx]].min(),iris_df[features[f_idx]].max(), 1000)
#     likelihood = cal_likelihood(x, iris_mean, iris_std)