import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_PATH = "../data/Iris.csv"


def get_gaussian_dist(x, mu, sigma):
    coefficient = 1 / np.sqrt(2 * np.pi * (sigma ** 2))
    exponent = np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2))
    return coefficient * exponent


def classify_data(test_data, te_likely_by_cls):
    te_df = pd.DataFrame(te_likely_by_cls)
    cls_names = [name for name, _ in te_df[0]]
    te_df = te_df.applymap(lambda x: x[1])
    simp_posterior = te_df.apply(np.prod)
    print(f"simp_posterior of test data: \n {simp_posterior.map('{:,.2f}'.format)}")
    max_idx = np.argmax(simp_posterior)
    prod = cls_names[max_idx]
    print(f"test data({test_data.to_list()}) is classified as {prod}")


def get_grid_data(x_min, x_max, x_mean_std_by_cls):
    x_grid = np.linspace(x_min, x_max, 1000)
    y_grid = list()
    for x in x_grid:
        curr = [get_gaussian_dist(x, mean, std) for i, mean, std in x_mean_std_by_cls]
        y_grid.append(curr)

    y_grid = pd.DataFrame(y_grid)
    return x_grid, y_grid


def plot_gaussian_distribution(idx, col, x_grid, y_grid, te_likely_by_cls, test_data, axes):
    for i in range(y_grid.shape[1]):
        axes[idx].plot(x_grid, y_grid.iloc[:, i])

    for cls, likelihood in te_likely_by_cls[idx]:
        axes[idx].scatter(test_data[idx], likelihood, label=f"{cls} = {likelihood:.3f}")

    axes[idx].set_title(col, weight='bold', fontsize=20)
    axes[idx].legend()


def gaussian_naive_bayes():
    df = pd.read_csv(DATA_FILE_PATH, index_col=0)

    # separate test data
    test_data = df.iloc[0, :]
    y_name = df.columns[-1]
    te_likely_by_cls = list()

    # figures setting
    fig, axes = plt.subplots(nrows=4, figsize=(10, 15))

    # feature iteration
    for idx, col in enumerate(df.columns.to_list()[:-1]):
        # subsets
        curr_df = df[[col, y_name]]
        y_uniques = curr_df.iloc[:, -1].unique()
        x_by_cls = [curr_df[curr_df[y_name] == y] for y in y_uniques]

        # calculate mean and std by class
        x_mean_std_by_cls = [(c.iloc[0, 1], c.iloc[:, 0].mean(), c.iloc[:, 0].std()) for c in x_by_cls]
        print(f"mean and std of {col}: ", x_mean_std_by_cls)

        # gaussian distribution(likelihood) for test data
        te_gs_by_cls = [(cls, get_gaussian_dist(test_data[col], mu, sigma)) for cls, mu, sigma in x_mean_std_by_cls]
        te_likely_by_cls.append(te_gs_by_cls)

        # linspace data for plotting
        curr_min = curr_df.iloc[:, 0].min()
        curr_max = curr_df.iloc[:, 0].max()
        x_grid, y_grid = get_grid_data(curr_min, curr_max, x_mean_std_by_cls)

        # plot gaussian distribution
        plot_gaussian_distribution(idx, col, x_grid, y_grid, te_likely_by_cls, test_data, axes)

    # classify test data
    classify_data(test_data, te_likely_by_cls)
    fig.tight_layout()



if __name__ == '__main__':
    gaussian_naive_bayes()
    plt.show()


