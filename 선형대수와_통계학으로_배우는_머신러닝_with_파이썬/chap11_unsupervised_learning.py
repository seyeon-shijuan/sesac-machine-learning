from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


def k_means_clustering_test():
    X, y = make_blobs(n_samples=100, n_features=2, centers=5, random_state=10)
    print(X.shape, y.shape)

    plt.scatter(X[:, 0], X[:,1], c='gray', edgecolor='black', marker='o')
    # plt.show()

    kmc = KMeans(n_clusters=5, init='random', max_iter=100, random_state=0)
    kmc.fit(X)
    label_kmc = kmc.labels_
    kmc_colums = ['kmc_comp1', 'kmc_comp2']
    X_kmc_df = pd.DataFrame(X, columns=kmc_colums)
    X_kmc_df['target'] = y
    X_kmc_df['label_kmc'] = label_kmc
    print(X_kmc_df.head(5))

    print(set(X_kmc_df['target']))
    print(set(X_kmc_df['label_kmc']))

    df = X_kmc_df
    markers = ['o', 'x', '^', 's', '*']
    for i, mark in enumerate(markers):
        df_i = df[df['label_kmc'] == i]
        target_i = i
        X1 = df_i['kmc_comp1']
        X2 = df_i['kmc_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)


    plt.xlabel('kmc_component1')
    plt.ylabel('kmc_component2')
    plt.legend()
    # plt.show()

    df = X_kmc_df
    markers = ['o', 'x', '^', 's', '*']
    for i, mark in enumerate(markers):
        df_i = df[df['target'] == i]
        target_i = i
        X1 = df_i['kmc_comp1']
        X2 = df_i['kmc_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)


    plt.xlabel('kmc_component1')
    plt.ylabel('kmc_component2')
    plt.legend()
    # plt.show()

    sil_score = silhouette_score(X, label_kmc)
    print(sil_score)


def agglomerative_clustering_test():
    # 계층 클러스터링
    X, y = make_blobs(n_samples=10, n_features=2, random_state=0)

    aggc = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='complete')
    label_aggc = aggc.fit_predict(X)
    print(label_aggc)

    aggc2 = AgglomerativeClustering(n_clusters=2, linkage='complete')
    label_aggc2 = aggc2.fit_predict(X)
    print(label_aggc2)

    aggc3 = AgglomerativeClustering(n_clusters=3, linkage='complete')
    label_aggc3 = aggc3.fit_predict(X)
    print(label_aggc3)

    linked = linkage(X, 'complete')
    labels = label_aggc
    dendrogram(linked, orientation='top', labels=labels, show_leaf_counts=True)
    plt.show()


def dbscan_test():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
    print(X.shape, y.shape)

    plt.scatter(X[:, 0], X[:, -1], c='gray', edgecolor='black', marker='o')
    # plt.show()

    dbs = DBSCAN(eps=0.2)
    dbs.fit(X)
    label_dbs = dbs.labels_
    print(label_dbs)

    dbs_columns = ['dbs_comp1', 'dbs_comp2']
    X_dbs_df = pd.DataFrame(X, columns=dbs_columns)
    X_dbs_df['target'] = y
    X_dbs_df['label_dbs'] = label_dbs
    print(X_dbs_df.head(5))

    print(set(X_dbs_df['target']))
    print(set(X_dbs_df['label_dbs']))


    df = X_dbs_df
    markers = ['o', 'x']

    for i, mark in enumerate(markers):
        df_i = df[df['label_dbs'] == i]
        target_i = i
        X1 = df_i['dbs_comp1']
        X2 = df_i['dbs_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)


    plt.xlabel('dbs_component1')
    plt.ylabel('dbs_component2')
    plt.legend()
    # plt.show()


    # 실제 타깃
    df = X_dbs_df
    markers = ['o', 'x']

    for i, mark in enumerate(markers):
        df_i = df[df['target'] == i]
        target_i = i
        X1 = df_i['dbs_comp1']
        X2 = df_i['dbs_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)


    plt.xlabel('dbs_component1')
    plt.ylabel('dbs_component2')
    plt.legend()
    # plt.show()

    sil_score = silhouette_score(X, label_dbs)
    print(sil_score)


def gaussian_mixture_test():
    X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=2)
    print(X.shape, y.shape)
    plt.scatter(X[:, 0], X[:, 1], c='gray', edgecolors='black', marker='o')
    # plt.show()

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X)
    label_gmm = gmm.predict(X)
    print(label_gmm)

    gmm_columns = ['gmm_comp1', 'gmm_comp2']
    X_gmm_df = pd.DataFrame(X, columns=gmm_columns)
    X_gmm_df['target'] = y
    X_gmm_df['label_gmm'] = label_gmm
    print(X_gmm_df.head(5))

    print(set(X_gmm_df['target']))
    print(set(X_gmm_df['label_gmm']))

    df = X_gmm_df
    markers=['o', 'x']

    for i, mark in enumerate(markers):
        df_i = df[df['label_gmm'] == i]
        target_i = i
        X1 = df_i['gmm_comp1']
        X2 = df_i['gmm_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)

    plt.xlabel('gmm_component1')
    plt.ylabel('gmm_component2')
    plt.legend()
    # plt.show()

    print(set(X_gmm_df['target']))
    print(set(X_gmm_df['label_gmm']))


    df = X_gmm_df
    markers = ['o', 'x']

    for i, mark in enumerate(markers):
        df_i = df[df['label_gmm'] == i]
        target_i = i
        X1 = df_i['gmm_comp1']
        X2 = df_i['gmm_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)


    plt.xlabel('gmm_component1')
    plt.ylabel('gmm_component2')
    plt.legend()
    # plt.show()


    # 실제 타깃
    df = X_gmm_df
    markers = ['o', 'x']

    for i, mark in enumerate(markers):
        df_i = df[df['target'] == i]
        target_i = i
        X1 = df_i['gmm_comp1']
        X2 = df_i['gmm_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)


    plt.xlabel('gmm_component1')
    plt.ylabel('gmm_component2')
    plt.legend()
    # plt.show()

    sil_score = silhouette_score(X, label_gmm)
    print(sil_score)


if __name__ == '__main__':
    gaussian_mixture_test()