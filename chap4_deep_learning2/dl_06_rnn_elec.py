import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import FinanceDataReader as fdr


df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False,
                 parse_dates={'dt': ['Date', 'Time']}, index_col='dt', dayfirst=True, na_values=['?'])

print(df.head())

df_resample = df.resample('h').mean()
raw_data = df_resample.dropna()
print(f"{raw_data.isnull().sum()=}")





