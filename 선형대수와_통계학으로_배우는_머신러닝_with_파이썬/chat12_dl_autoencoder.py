import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation


np.random.seed(0)
tf.random.set_seed(0)

(X_tn0, y_tn0), (X_te0, y_te0) = datasets.mnist.load_data()

print(f"X_tn0.shape = {X_tn0.shape}")
print(f"y_tn0.shape = {y_tn0.shape}")
print(f"X_te0.shape = {X_te0.shape}")
print(f"y_te0.shape = {y_te0.shape}")

# plt.figure(figsize=(10, 5))
# for i in range(2*5):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(X_tn0[i].reshape((28, 28)), cmap='Greys')

# plt.show()

X_tn_re = X_tn0.reshape(60000, 28, 28, 1)
X_tn = X_tn_re / 255
X_te_re = X_te0.reshape(10000, 28, 28, 1)
X_te = X_te_re / 255

print(f"X_tn.shape = {X_tn.shape}, X_te.shape = {X_te.shape}")

# 노이즈 피처 데이터
X_tn_noise = X_tn + np.random.uniform(-1, 1, size=X_tn.shape)
X_te_noise = X_te + np.random.uniform(-1, 1, size=X_te.shape)

print(f"X_tn_noise.shape = {X_tn_noise.shape}")

X_tn_ns = np.clip(X_tn_noise, a_min=0, a_max=1)
X_te_ns = np.clip(X_te_noise, a_min=0, a_max=1)

# plt.figure(figsize=(10, 5))
# for i in range(2*5):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(X_tn_ns[i].reshape((28, 28)), cmap='Greys')
#
# plt.show()

# 인코더
input_layer1 = Input(shape=(28, 28, 1))
tmp = Conv2D(20, kernel_size=(5, 5), padding="same")
tmp2 = tmp(input_layer1)

x1 = Conv2D(20, kernel_size=(5, 5), padding="same")(input_layer1)
x1 = Activation(activation="relu")(x1)
