import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
plt.switch_backend('newbackend')

# 집값 예측
raw_boston = datasets.load_boston()
X_boston = pd.DataFrame(raw_boston.data)
Y_boston = pd.DataFrame(raw_boston.target)
df_boston = pd.concat([X_boston, Y_boston], axis=1)

print(1)
