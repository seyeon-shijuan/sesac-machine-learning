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


def kmeans_plusplus_init(X, k):
    init_cent = X[np.random.choice(len(X))]
    centers = [init_cent]

    print('here')

    for _ in range(1, k):
        near_dists = np.array([min([np.linalg.norm(x - c)**2 for c in centers]) for x in X])
        probs = near_dists / near_dists.sum()
        next_cent = X[np.random.choice(len(X), p=probs)]
        centers.append(next_cent)

    return centers


def kmeans_clustering(X, centers, max_iters=100):
    for _ in range(max_iters):
        labels = np.array([np.argmin([np.linalg.norm(x - c)**2 for c in centers]) for x in X])

        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(len(centers))])

        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels


def kmeans_visualization(X, y, cents):
    class_colors = ['#FF5733', '#FFA500', '#008000', '#FF69B4']

    fig, ax = plt.subplots(figsize=(7, 7))

    # all data
    ax.scatter(x=X[:, 0], y=X[:, 1], c=[class_colors[label] for label in y], alpha=0.5)

    # centroids
    for c in cents:
        ax.scatter(x=c[0], y=c[1], color="b")

    plt.show()



if __name__ == '__main__':
    data, target_cls, centroids = get_sample_dataset(n_classes=4, n_data=100)
    centers = kmeans_plusplus_init(X=data, k=4)
    final_centers, labels = kmeans_clustering(X=data, centers=centers)
    kmeans_visualization(data, labels, final_centers)



