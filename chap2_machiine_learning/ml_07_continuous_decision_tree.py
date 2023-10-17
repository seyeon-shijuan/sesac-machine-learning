import pandas as pd
import numpy as np


def get_variance_idx(X, y, col_names):
    var_by_column_list = list()

    # column 순서 대로 순회
    for i in range(X.shape[1]):
        x = X[:, i]
        # 인덱스 종류 구하기
        uniques = np.unique(x)
        mean_list = list()
        var_list = list()
        weight_list = list()

        # 인덱스 종류별로 분산 구하기
        for item in uniques:
            idx = np.where(x == item)
            tmp = np.array(y[idx]).astype(dtype=float)
            curr_mean = np.mean(tmp)
            curr_var = np.var(tmp)
            curr_weight = len(idx) / len(x)

            mean_list.append(curr_mean)
            var_list.append(curr_var)
            weight_list.append(curr_weight)

        # 해당 column의 가중평균을 적용한 최종 분산값 구하기
        var_list = np.array(var_list)
        weight_list = np.array(weight_list)
        fin_val = np.sum(var_list * weight_list)
        var_by_column_list.append(fin_val)

    min_idx = np.argmin(var_by_column_list)
    print("argmin(variance) column in current layer: ", col_names[min_idx])

    return min_idx


def continuous_decision_tree():
    df = pd.read_csv('../data/season.csv')
    data = df.to_numpy().tolist()
    # print(data)
    data = [[1, 'winter', False, 800],
            [2, 'winter', False, 826],
            [3, 'winter', True, 900],
            [4, 'spring', False, 2100],
            [5, 'spring', True, 4740],
            [6, 'spring', True, 4900],
            [7, 'summer', False, 3000],
            [8, 'summer', True, 5800],
            [9, 'summer', True, 6200],
            [10, 'autumn', False, 2910],
            [11, 'autumn', False, 2880],
            [12, 'autumn', True, 2820]]

    data = np.array(data)

    X = data[:, 1:-1]
    y = data[:, -1]
    col_names = ['SEASON', 'DAY']

    min_idx = get_variance_idx(X, y, col_names)
    print(f"{min_idx = }")

    # SEASON
    spring = np.where(X[:, 0] == "spring")
    summer = np.where(X[:, 0] == "summer")
    autumn = np.where(X[:, 0] == "autumn")
    winter = np.where(X[:, 0] == "winter")

    X_spring = X[spring, -1]
    X_summer = X[summer, -1]
    X_autumn = X[autumn, -1]
    X_winter = X[winter, -1]

    y_spring = y[spring]
    y_summer = y[summer]
    y_autumn = y[autumn]
    y_winter = y[winter]

    h1_spring_idx = get_variance_idx(X_spring, y_spring, col_names=['DAY'])
    h1_summer_idx = get_variance_idx(X_summer, y_summer, col_names=['DAY'])
    h1_autumn_idx = get_variance_idx(X_autumn, y_autumn, col_names=['DAY'])
    h1_winter_idx = get_variance_idx(X_winter, y_winter, col_names=['DAY'])


if __name__ == '__main__':
    continuous_decision_tree()

