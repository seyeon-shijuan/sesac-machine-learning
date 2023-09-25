from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

# 지도 학습

def knn_test():
    raw_iris = datasets.load_iris()

    X = raw_iris.data
    y = raw_iris.target

    # random_state = random seed
    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

    # standardization
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    # train
    clf_knn = KNeighborsClassifier(n_neighbors=2)
    clf_knn.fit(X_tn_std, y_tn)

    # predict
    knn_pred = clf_knn.predict(X_te_std)

    # accuracy
    accuracy = accuracy_score(y_te, knn_pred)

    # confusion matrix
    conf_matrix = confusion_matrix(y_te, knn_pred)
    print(conf_matrix)

    # classification report
    class_report = classification_report(y_te, knn_pred)
    print(class_report)


def linear_regression_test():
    # 선형 회귀 분석 실습

    raw_boston = datasets.load_boston()

    X = raw_boston.data
    y = raw_boston.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=1)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_lr = LinearRegression()
    clf_lr.fit(X_tn_std, y_tn)
    print(clf_lr.coef_) # 추정된 회귀 계수 확인
    print(clf_lr.intercept_) # 추정된 상수항 확인

    # ridge lasso

    clf_ridge = Ridge(alpha=1)
    clf_ridge.fit(X_tn_std, y_tn)
    print("ridge coef: ",clf_ridge.coef_)
    print("ridge intercept: ",clf_ridge.intercept_)

    clf_lasso = Lasso(alpha=0.01)
    clf_lasso.fit(X_tn_std, y_tn)
    print("lasso coef: ",clf_lasso.coef_)
    print("lasso intercept: ",clf_lasso.intercept_)

    # elasticnet
    clf_elastic = ElasticNet(alpha=0.01, l1_ratio=0.01)
    clf_elastic.fit(X_tn_std, y_tn)
    # l1_ratio가 0이면 L2제약만 적용
    print("Elasticnet coef: ",clf_elastic.coef_)
    print("Elasticnet intercept: ",clf_elastic.intercept_)

    # 데이터 예측
    pred_lr = clf_lr.predict(X_te_std)
    pred_ridge = clf_ridge.predict(X_te_std)
    pred_lasso = clf_lasso.predict(X_te_std)
    pred_elastic = clf_elastic.predict(X_te_std)

    # 모형 평가 R 제곱값
    print("r2 score [linear]: ", r2_score(y_te, pred_lr))
    print("r2 score [ridge]: ", r2_score(y_te, pred_ridge))
    print("r2 score [lasso]: ", r2_score(y_te, pred_lasso))
    print("r2 score [elastic]: ", r2_score(y_te, pred_elastic))
    # r2 score은 높을 수록 좋은 것임

    # 모형평가 MSE
    print("MSE [linear]: ", mean_squared_error(y_te, pred_lr))
    print("MSE [ridge]: ", mean_squared_error(y_te, pred_ridge))
    print("MSE [lasso]: ", mean_squared_error(y_te, pred_lasso))
    print("MSE [elastic]: ", mean_squared_error(y_te, pred_elastic))
    # MSE는 오차값이라서 작을 수록 좋은 것임


def logistic_regression_test():
    raw_cancer = datasets.load_breast_cancer()
    X = raw_cancer.data
    y = raw_cancer.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_logi_l2 = LogisticRegression(penalty='l2')
    clf_logi_l2.fit(X_tn_std, y_tn)

    print(clf_logi_l2.coef_)
    print(clf_logi_l2.intercept_)

    # data prediction
    pred_logistic = clf_logi_l2.predict(X_te_std)
    print(pred_logistic)
    pred_proba = clf_logi_l2.predict_proba(X_te_std)
    print(pred_proba)

    # validation
    precision = precision_score(y_te, pred_logistic)
    print(precision)
    conf_matrix = confusion_matrix(y_te, pred_logistic)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_logistic)
    print(class_report)


def naive_bayes_test():
    # 나이브 베이즈
    raw_wine = datasets.load_wine()

    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_gnb = GaussianNB()
    clf_gnb.fit(X_tn_std, y_tn)
    pred_gnb = clf_gnb.predict(X_te_std)
    print(pred_gnb)

    recall = recall_score(y_te, pred_gnb, average='macro')
    print(recall)

    conf_matrix = confusion_matrix(y_te, pred_gnb)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_gnb)
    print(class_report)


def decision_tree_test():
    # decision tree
    raw_wine = datasets.load_wine()
    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_tree = tree.DecisionTreeClassifier(random_state=0)
    clf_tree.fit(X_tn_std, y_tn)
    pred_tree = clf_tree.predict(X_te_std)
    print(pred_tree)

    f1 = f1_score(y_te, pred_tree, average='macro')
    print(f1)

    conf_matrix = confusion_matrix(y_te, pred_tree)
    print(conf_matrix)

    class_report = classification_report(y_te, pred_tree)
    print(class_report)


def svm_test():
    # support vector machine
    raw_wine = datasets.load_wine()
    X = raw_wine.data
    y = raw_wine.target
    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_svm_lr = svm.SVC(kernel='linear', random_state=0)
    clf_svm_lr.fit(X_tn_std, y_tn)
    pred_svm = clf_svm_lr.predict(X_te_std)
    print(pred_svm)

    accuracy = accuracy_score(y_te, pred_svm)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_svm)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_svm)
    print(class_report)


def cross_validation_test():
    # cross validation
    raw_wine = datasets.load_wine()
    X = raw_wine.data
    y = raw_wine.target
    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)


    param_grid = {'kernel': ('linear', 'rbf'),
                  'C': [0.5, 1, 10, 100]}

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    svc = svm.SVC(random_state=0)
    grid_cv = GridSearchCV(svc, param_grid, cv=kfold, scoring='accuracy')
    grid_cv.fit(X_tn_std, y_tn)

    print(np.transpose(pd.DataFrame(grid_cv.cv_results_)))

    print(grid_cv.best_score_)
    print(grid_cv.best_params_)

    clf = grid_cv.best_estimator_
    print("clf: ", clf)

    # 방법 1
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_scores = cross_validate(clf, X_tn_std, y_tn, cv=kfold, scoring=metrics)
    print(cv_scores)

    # 방법 2
    cv_score = cross_val_score(clf, X_tn_std, y_tn, cv=kfold, scoring='accuracy')
    print("cv score:", cv_score)
    print("mean: ",cv_score.mean())
    print("std: ", cv_score.std())

    pred_svm = clf.predict(X_te_std)
    print(pred_svm)

    accuracy = accuracy_score(y_te, pred_svm)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_svm)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_svm)
    print(class_report)


if __name__ == '__main__':
    # knn_test()
    # linear_regression_test()
    # logistic_regression_test()
    # naive_bayes_test()
    # decision_tree_test()
    # svm_test()
    cross_validation_test()
