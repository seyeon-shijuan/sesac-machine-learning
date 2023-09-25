import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

# 머신러닝 데이터 살펴보기

''' 데이터셋 설명 '''
'''
# 집값 예측
raw_boston = datasets.load_boston()
X_boston = pd.DataFrame(raw_boston.data)
y_boston = pd.DataFrame(raw_boston.target)
df_boston = pd.concat([X_boston, y_boston], axis=1)

print(len(df_boston))

feature_boston = raw_boston.feature_names
col_boston = np.append(feature_boston, ['target'])
df_boston.columns = col_boston

# 아이리스 데이터(datasets.load_iris())
raw_iris = datasets.load_iris()
X_iris = pd.DataFrame(raw_iris.data)
Y_iris = pd.DataFrame(raw_iris.target)
df_iris = pd.concat([X_iris, Y_iris], axis=1)

feature_iris = raw_iris.feature_names
col_iris = np.append(feature_iris, ['target'])
df_iris.columns = col_iris

# 와인 데이터(load_wine())
raw_wine = datasets.load_wine()
X_wine = pd.DataFrame(raw_wine.data)
Y_wine = pd.DataFrame(raw_wine.target)
df_wine = pd.concat([X_wine, Y_wine], axis=1) # axis 1은 옆으로 붙이기

feature_wine = raw_wine.feature_names
col_wine = np.append(feature_wine, ['target'])
df_wine.columns = col_wine


# 당뇨병 데이터(datasets.load_diabetes())
raw_diab = datasets.load_diabetes()
X_diab = pd.DataFrame(raw_diab.data)
Y_diab = pd.DataFrame(raw_diab.target)
df_diab = pd.concat([X_diab, Y_diab], axis=1)
feature_diab = raw_diab.feature_names
col_diab = np.append(feature_diab, ['target'])
df_diab.columns = col_diab

# 유방암 데이터(datasets.load_breast_cancer()
raw_bc = datasets.load_breast_cancer()
X_bc = pd.DataFrame(raw_bc.data)
y_bc = pd.DataFrame(raw_bc.target)
df_bc = pd.concat([X_bc,y_bc], axis=1)
feature_bc = raw_bc.feature_names
col_bc = np.append(feature_bc, ['target'])
df_bc.columns = col_bc

'''

''' 결측치처리_클래스레이블_원핫 '''

# 결측치 처리
df = pd.DataFrame([
    [42, 'male', 12,'reading','class2'],
    [35, 'unknown', 3,'cooking', 'class1'],
    [1000, 'female', 7,'cycling', 'class3'],
    [1000, 'unknown', 21,'unknown', 'unknown']
])
df.columns = ['age', 'gender', 'month_birth', 'hobby',  'target']
df['age'].unique()
df['gender'].unique()
df['month_birth'].unique()
df['hobby'].unique()
df['target'].unique()

df.loc[df['age'] > 150, ['age']] = np.nan
df.loc[df['gender'] == 'unknown', ['gender']] = np.nan
df.loc[df['month_birth'] >12, ['month_birth']] = np.nan
df.loc[df['hobby'] == 'unknown', ['hobby']] = np.nan
df.loc[df['target'] == 'unknown', ['target']] = np.nan

# 결측치 포함한 row 삭제
df2 = df.dropna(axis=0)
df3 = df.dropna(axis=1)
# 모든 값이 결측치인 행 삭제
df4 = df.dropna(how='all')
df5 = df.dropna(thresh=2)
# 특정 열에 결측치가 있는 경우 record 삭제
df6 = df.dropna(subset=['gender'])
df.isnull().sum()

# 결측치 대체하기
alter_values = {'age': 0,
                'gender': 'U',
                'month_birth': 0,
                'hobby': 'U',
                'target': 'class4'}
df7 = df.fillna(value=alter_values)

# 클래스 라벨 설정
df8 = df7
class_label = LabelEncoder()
data_value = df8['target'].values
y_new = class_label.fit_transform(data_value)
df8['target'] = y_new

y_ori = class_label.inverse_transform(y_new)
df8['target'] = y_ori

# 클래스 라벨링 without 라이브러리
y_arr = df8['target'].values
y_arr.sort()
num_y = 0
dic_y = {}
for ith_y in y_arr:
    dic_y[ith_y] = num_y
    num_y += 1

df8['target'] = df8['target'].replace(dic_y)

# 원-핫 인코딩
df9 = df8
df9['target'] = df9['target'].astype(str)
df10 = pd.get_dummies(df9['target'])
df9['target'] = df9['target'].astype
df11 = pd.get_dummies(df9['target'], drop_first=True)
df12 = df8
df13 = pd.get_dummies(df12)
df14 = pd.get_dummies(df12, drop_first=True)

# 두번째 방법
hot_encoder = OneHotEncoder()
y = df7[['target']]
y_hot = hot_encoder.fit_transform(y)

# tensorflow로 원핫인코딩
# y_hotec = to_categorical(y)
# print(y_hotec)

# 표준화 스케일링
std = StandardScaler()
tmp = df8[['month_birth']]
std.fit(df8[['month_birth']])
x_std = std.transform(df8[['month_birth']])
x_std2 = std.fit_transform(df8[['month_birth']])

np.mean(x_std)
np.std(x_std)

# 로버스트 스케일링
robust = RobustScaler()
robust.fit(df8[['month_birth']])
x_robust = robust.transform(df8[['month_birth']])

# 최소-최대 스케일링
minmax = MinMaxScaler()
minmax.fit(df8[['month_birth']])
x_minmax = minmax.transform(df8[['month_birth']])

# 노멀 스케일링
normal = Normalizer()
normal.fit(df8[['month_birth']])
x_normal = normal.transform(df8[['age', 'month_birth']])


print('here')

