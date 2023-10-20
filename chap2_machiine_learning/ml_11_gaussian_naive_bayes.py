import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_gaussian_dist(x, mu, sigma):
    coefficient = 1 / np.sqrt(2 * np.pi * (sigma ** 2))
    exponent = np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2))
    return coefficient * exponent


def gaussian_naive_bayes():
    df = pd.read_csv("../data/Iris.csv", index_col=0)

    # separate test data
    test_data = df.iloc[0, :]
    y_name = df.columns[-1]
    te_likely_by_cls = list()


    # figures setting
    # fig, axes = plt.subplots(nrows=4, figsize=(15, 5))

    # Feature Iteration
    for col in df.columns.to_list()[:-1]:
        # subsets
        curr_df = df[[col, y_name]]
        y_uniques = curr_df.iloc[:, -1].unique()
        x_by_cls = [curr_df[curr_df[y_name] == y] for y in y_uniques]

        # mean, std by class
        x_mean_std_by_cls = [(c.iloc[0, 1], c.iloc[:, 0].mean(), c.iloc[:, 0].std()) for c in x_by_cls]
        print(f"{col}: ", x_mean_std_by_cls)

        # get val from gaussian distribution formula
        x_gs_by_cls = list()
        min_max_by_cls = pd.DataFrame(columns=['min', 'max'])

        for i, c in enumerate(x_by_cls):
            # initial val for x axis
            curr_min = c.iloc[:, 0].min()
            curr_max = c.iloc[:, 0].max()
            min_max_by_cls.loc[i] = [curr_min, curr_max]

            curr = c.copy()
            # gaussian distribution val for y axis
            curr[c.columns[0]] = c.iloc[:, 0].apply(get_gaussian_dist, args=(x_mean_std_by_cls[i][1], x_mean_std_by_cls[i][2]))
            x_gs_by_cls.append(curr)

        # concatenate by cls
        concat_cls = list()
        for x1, x2 in zip(x_by_cls, x_gs_by_cls):
            curr_cls = pd.concat([x1.iloc[:, 0], x2.iloc[:, :]], axis=1)
            curr_cls.columns.values[1] = 'Gaussian'
            concat_cls.append(curr_cls)

        # gaussian distribution(likelihood) for test data
        te_gs_by_cls = [(cls, get_gaussian_dist(test_data[col], mu, sigma)) for cls, mu, sigma in x_mean_std_by_cls]
        te_likely_by_cls.append(te_gs_by_cls)
        # prod = np.prod([gs for cls, gs in te_gs_by_cls])

        # linspace data for plotting
        x_min = min_max_by_cls.iloc[:, 0].min()
        x_max = min_max_by_cls.iloc[:, 1].max()

        x_grid = np.linspace(x_min, x_max, 1000)


        print('here')




    # classify the test data
    te_df = pd.DataFrame(te_likely_by_cls)
    cls_names = [name for name, _ in te_df[0]]
    te_df = te_df.applymap(lambda x: x[1])
    simp_posterior = te_df.apply(np.prod)
    print(f"simp_posterior: \n {simp_posterior.map('{:,.2f}'.format)}")
    max_idx = np.argmax(simp_posterior)

    prod = cls_names[max_idx]
    print(f"test data({test_data.to_list()}) is classified as {prod}")




if __name__ == '__main__':
    gaussian_naive_bayes()


