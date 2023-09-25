from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# 모형 평가

raw_boston = datasets.load_boston()

X = raw_boston.data
y = raw_boston.target

# 트레이닝 / 테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=7)

# 표준화 스케일링
std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 학습
clf_linear = LinearRegression()
clf_linear.fit(X_tn_std, y_tn)

# 예측
pred_linear = clf_linear.predict(X_te_std)

# 평가
print(mean_squared_error(y_te, pred_linear))

# 파이프라인

# 트레이닝 / 테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=7)

# 파이프라인
linear_pipline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
])

# 학습
linear_pipline.fit(X_tn, y_tn)

# 예측
pred_linear = linear_pipline.predict(X_te)

# 평가
print(mean_squared_error(y_te, pred_linear))


# 그리드 서치
# 꽃 데이터 불러오기
raw_iris = datasets.load_iris()

# 피쳐 / 타겟
X = raw_iris.data
y = raw_iris.target

# 트레이닝 / 테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

# 표준화 스케일
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

best_accuracy = 0

for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X_tn_std, y_tn)
    knn_pred = clf_knn.predict(X_te_std)
    accuracy = accuracy_score(y_te, knn_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        final_k = {'k': k}


print(final_k)
print(accuracy)


# 분류(정확도)
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred, normalize=False))

# confusion matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))



