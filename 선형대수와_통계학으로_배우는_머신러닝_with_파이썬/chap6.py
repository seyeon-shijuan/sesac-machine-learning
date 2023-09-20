import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# 집값 예측
raw_boston = datasets.load_boston()
X_boston = pd.DataFrame(raw_boston.data)
y_boston = pd.DataFrame(raw_boston.target)
df_boston = pd.concat([X_boston, y_boston], axis=1)

print(len(df_boston))


