# https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
import numpy as np
import matplotlib.pyplot as plt


def get_data_from_centroid(centroid, n_data):
    data = np.random.normal(loc=centroid, scale=1, size=(n_data, 2))
    return data


def get_sample_dataset(n_classes, n_data):
    centroids = np.array([np.random.uniform(low=-10, high=10, size=(2,)) for x in range(n_classes)])
    target_cls = np.array([i for i in range(n_classes) for _ in range(n_data)])
    data = None

    for i, centroid in enumerate(centroids):
        if i == 0:
            data = get_data_from_centroid(centroid, n_data)
            continue

        curr_dataset = get_data_from_centroid(centroid, n_data)
        data = np.vstack([data, curr_dataset])

    return data, target_cls, centroids


if __name__ == '__main__':
    pass



