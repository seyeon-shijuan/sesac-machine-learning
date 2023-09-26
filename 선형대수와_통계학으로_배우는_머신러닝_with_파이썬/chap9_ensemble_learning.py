from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def voting_test():
    raw_iris = datasets.load_iris()
    X = raw_iris.data
    y = raw_iris.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = svm.SVC(kernel='linear', random_state=1)
    clf3 = GaussianNB()

    clf_voting = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('svm', clf2),
            ('gnb', clf3)
        ],
        voting='hard',
        weights=[1, 1, 1]
    )
    # hard voting = 과반수

    clf_voting.fit(X_tn_std, y_tn)

    pred_voting = clf_voting.predict(X_te_std)
    print(pred_voting)
    accuracy = accuracy_score(y_te, pred_voting)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_voting)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_voting)
    print(class_report)


def random_forest_test():
    raw_wine = datasets.load_wine()
    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf.fit(X_tn_std, y_tn)

    pred_rf = clf_rf.predict(X_te_std)
    print(pred_rf)

    accuracy = accuracy_score(y_te, pred_rf)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_rf)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_rf)
    print(class_report)


def bagging_test():
    raw_wine = datasets.load_wine()
    X = raw_wine.data
    y = raw_wine.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    # base_estimator=개별학습기, n_estimators=학습기 개수
    clf_bagging = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, random_state=0)
    clf_bagging.fit(X_tn_std, y_tn)

    pred_bagging = clf_bagging.predict(X_te_std)
    print(pred_bagging)
    accuracy = accuracy_score(y_te, pred_bagging)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_bagging)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_bagging)
    print(class_report)


def ada_boost_test():
    # 에이다 부스트
    raw_breast_cancer = datasets.load_breast_cancer()
    X = raw_breast_cancer.data
    y = raw_breast_cancer.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_ada = AdaBoostClassifier(random_state=0)
    clf_ada.fit(X_tn_std, y_tn)

    pred_ada = clf_ada.predict(X_te_std)
    print(pred_ada)
    accuracy = accuracy_score(y_te, pred_ada)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_ada)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_ada)
    print(class_report)


def gradient_boosting_test():
    raw_breast_cancer = datasets.load_breast_cancer()
    X = raw_breast_cancer.data
    y = raw_breast_cancer.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf_gbt = GradientBoostingClassifier(max_depth=2, learning_rate=0.01, random_state=0)
    clf_gbt.fit(X_tn_std, y_tn)

    pred_gboost = clf_gbt.predict(X_te_std)
    print(pred_gboost)
    accuracy = accuracy_score(y_te, pred_gboost)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_gboost)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_gboost)
    print(class_report)


def stacking_test():
    # stacking
    raw_breast_cancer = datasets.load_breast_cancer()
    X = raw_breast_cancer.data
    y = raw_breast_cancer.target

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)
    std_scale = StandardScaler()
    std_scale.fit(X_tn)
    X_tn_std = std_scale.transform(X_tn)
    X_te_std = std_scale.transform(X_te)

    clf1 = svm.SVC(kernel='linear', random_state=1)
    clf2 = GaussianNB()

    # estimators=기본 학습기, final_estimator=메타 학습기
    clf_stkg = StackingClassifier(
        estimators=[
            ('svm', clf1),
            ('gnb', clf2)
        ],
        final_estimator=LogisticRegression()
    )
    clf_stkg.fit(X_tn_std, y_tn)

    pred_stkg = clf_stkg.predict(X_te_std)
    print(pred_stkg)

    accuracy = accuracy_score(y_te, pred_stkg)
    print(accuracy)
    conf_matrix = confusion_matrix(y_te, pred_stkg)
    print(conf_matrix)
    class_report = classification_report(y_te, pred_stkg)
    print(class_report)


if __name__ == '__main__':
    stacking_test()