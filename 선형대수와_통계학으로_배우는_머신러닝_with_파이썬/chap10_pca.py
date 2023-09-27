from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt


def pca_test():
    raw_wine = datasets.load_wine()

    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=1)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    # n_components= 줄여서 만들 차원 수
    pca = PCA(n_components=2)
    pca.fit(X_tn_std)
    X_tn_pca = pca.transform(X_tn_std)
    X_te_pca = pca.transform(X_te_std)
    print(X_tn_std.shape) # (133, 13)
    print(X_tn_pca.shape) # (133, 2)

    print(pca.get_covariance())
    print("고윳값: ", pca.singular_values_)
    print(pca.components_) # 고유 벡터 -> 주성분 벡터

    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    pca_columns = ['pca_comp1', 'pca_comp2']
    X_tn_pca_df = pd.DataFrame(X_tn_pca, columns=pca_columns)
    X_tn_pca_df['target'] = y_tn
    print(X_tn_pca_df.head(5))

    plt.scatter(X_tn_pca_df['pca_comp1'], X_tn_pca_df['pca_comp2'], marker='o')
    plt.xlabel('pca_component1')
    plt.ylabel('pca_component2')
    # plt.show()

    # diff group
    df = X_tn_pca_df
    df_0 = df[df['target'] == 0]
    df_1 = df[df['target'] == 1]
    df_2 = df[df['target'] == 2]

    X_11 = df_0['pca_comp1']
    X_12 = df_1['pca_comp1']
    X_13 = df_2['pca_comp1']

    X_21 = df_0['pca_comp2']
    X_22 = df_1['pca_comp2']
    X_23 = df_2['pca_comp2']

    target_0 = raw_wine.target_names[0]
    target_1 = raw_wine.target_names[1]
    target_2 = raw_wine.target_names[2]

    plt.scatter(X_11, X_21, marker='o', label=target_0)
    plt.scatter(X_12, X_22, marker='x', label=target_1)
    plt.scatter(X_13, X_23, marker='^', label=target_2)

    plt.xlabel('pca_component1')
    plt.ylabel('pca_component2')
    plt.legend()
    # plt.show()

    # for loop
    df = X_tn_pca_df
    markers = ['o', 'x', '^']
    for i, mark in enumerate(markers):
        df_i = df[df['target'] == i]
        target_i = raw_wine.target_names[i]
        X1 = df_i['pca_comp1']
        X2 = df_i['pca_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)

    plt.xlabel('pca_component1')
    plt.ylabel('pca_component2')
    plt.legend()
    # plt.show()

    # pca 적용 이전 데이터 학습
    clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf.fit(X_tn_std, y_tn)
    pred_rf = clf_rf.predict(X_te_std)
    print(pred_rf)

    accuracy = accuracy_score(y_te, pred_rf)
    print(accuracy)

    clf_rf_pca = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf_pca.fit(X_tn_pca, y_tn)
    pred_rf_pca = clf_rf_pca.predict(X_te_pca)

    accuracy_pca = accuracy_score(y_te, pred_rf_pca)
    print(accuracy_pca)


def kernel_pca_test():
    # kernel pca
    raw_wine = datasets.load_wine()

    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=1)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    k_pca = KernelPCA(n_components=2, kernel='poly')
    # kernel은 linear, poly, rbf, sigmoid, cosine, precomputed 중에 선택 가능
    k_pca.fit(X_tn_std)
    X_tn_kpca = k_pca.transform(X_tn_std)
    X_te_kpca = k_pca.transform(X_te_std)
    print(X_tn_std.shape)
    print(X_tn_kpca.shape)

    print("lambdas 고윳값: ", k_pca.lambdas_)
    print("alphas 고유 벡터: ", k_pca.alphas_)

    kpca_columns = ['kpca_comp1', 'kpca_comp2']
    X_tn_kpca_df = pd.DataFrame(X_tn_kpca, columns=kpca_columns)
    X_tn_kpca_df['target'] = y_tn
    print(X_tn_kpca_df.head(5))

    df = X_tn_kpca_df
    markers = ['o', 'x', '^']

    for i, mark in enumerate(markers):
        X_i = df[df['target'] == i]
        target_i = raw_wine.target_names[i]
        X1 = X_i['kpca_comp1']
        X2 = X_i['kpca_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)

    plt.xlabel('kernel_pca_component1')
    plt.ylabel('kernel_pca_component2')

    plt.legend()
    plt.show()

    # 차원 축소된 데이터를 이용해 커널 PCA 학습
    clf_rf_kpca = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf_kpca.fit(X_tn_kpca, y_tn)
    pred_rf_kpca = clf_rf_kpca.predict(X_te_kpca)

    accuracy = accuracy_score(y_te, pred_rf_kpca)
    print(accuracy)


def lda_test():
    # LDA
    raw_wine = datasets.load_wine()

    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=1)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tn_std, y_tn)
    X_tn_lda = lda.transform(X_tn_std)
    X_te_lda = lda.transform(X_te_std)

    print(X_tn_std.shape)
    print(X_te_std.shape)
    print(lda.intercept_)
    print(lda.coef_)

    lda_columns = ['lda_comp1', 'lda_comp2']
    X_tn_lda_df = pd.DataFrame(X_tn_lda, columns=lda_columns)
    X_tn_lda_df['target'] = y_tn
    print(X_tn_lda_df.head(5))

    df = X_tn_lda_df
    markers = ['o', 'x', '^']

    for i, mark in enumerate(markers):
        X_i = df[df['target'] == i]
        target_i = raw_wine.target_names[i]
        X1 = X_i['lda_comp1']
        X2 = X_i['lda_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)

    plt.xlabel("lda_comp1")
    plt.ylabel("lda_comp2")
    plt.legend()
    # plt.show()

    clf_rf_lda = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf_lda.fit(X_tn_lda, y_tn)
    pred_rf_lda = clf_rf_lda.predict(X_te_lda)
    print(pred_rf_lda)

    accuracy = accuracy_score(y_te, pred_rf_lda)
    print(accuracy)


def lle_test():
    raw_wine = datasets.load_wine()

    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=1)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    lle = LocallyLinearEmbedding(n_components=2)
    lle.fit(X_tn_std, y_tn)
    X_tn_lle = lle.transform(X_tn_std)
    X_te_lle = lle.transform(X_te_std)

    print(X_tn_std.shape)
    print(X_tn_lle.shape)

    print(lle.embedding_)

    lle_columns = ['lle_comp1', 'lle_comp2']
    X_tn_lle_df = pd.DataFrame(X_tn_lle, columns=lle_columns)
    X_tn_lle_df['target'] = y_tn
    print(X_tn_lle_df.head(5))

    df = X_tn_lle_df
    markers = ['o', 'x', '^']

    for i, mark in enumerate(markers):
        X_i = df[df['target'] == i]
        target_i = raw_wine.target_names[i]
        X1 = X_i['lle_comp1']
        X2 = X_i['lle_comp2']
        plt.scatter(X1, X2, marker=mark, label=target_i)

    plt.xlabel("lle_comp1")
    plt.ylabel("lle_comp2")
    plt.legend()
    plt.show()

    clf_rf_lle = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf_lle.fit(X_tn_lle, y_tn)
    pred_rf_lle = clf_rf_lle.predict(X_te_lle)
    print(pred_rf_lle)

    accuracy = accuracy_score(y_te, pred_rf_lle)
    print(accuracy)


