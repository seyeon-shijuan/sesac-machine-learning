import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment
import os


DATA_DIR = "../data/"
data = pd.read_pickle(DATA_DIR + 'base_dataset_standardized.pkl') #불러오기

s = ClassificationExperiment()
s.setup(data, target=data.columns[-1], session_id=123)
print(s)

best = s.compare_models()
print(best)

s.evaluate_model(best)
s.plot_model(best, plot='auc')
s.plot_model(best, plot='confusion_matrix')
s.predict_model(best)



print('here')