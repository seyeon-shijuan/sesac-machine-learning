import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

(X_tn0, y_tn0), (X_te0, y_test) = datasets.imdb.load_data(num_words=2000)
print(X_tn0.shape)

X_train = X_tn0[0:20000]
y_train = y_tn0[0:20000]
X_valid = X_tn0[20000:25000]
y_valid = y_tn0[20000:25000]

X_train = sequence.pad_sequences(X_train, maxlen=100)
X_valid = sequence.pad_sequences(X_valid, maxlen=100)
X_test = sequence.pad_sequences(X_te0, maxlen=100)

# LSTM 모형 생성
model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=100))
# model.add(Conv1D(50, kernel_size=3, padding='valid', activation='relu'))
# model.add(MaxPooling1D(pool_size=3))
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 모형 컴파일 및 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=100, epochs=10, validation_data=(X_valid, y_valid))

# 모형 평가
print(f"model.evaluate(X_train, y_train)[1] : {model.evaluate(X_train, y_train)[1]}")
print(f"model.evaluate(X_valid, y_valid)[1] : {model.evaluate(X_valid, y_valid)[1]}")
print(f"model.evaluate(X_test, y_test)[1] : {model.evaluate(X_test, y_test)[1]}")

# 정확도 및 손실 그래프
epoch = np.arange(1, 11)
acc_train = hist.history['accuracy']
acc_valid = hist.history['val_accuracy']
loss_train = hist.history['loss']
loss_valid = hist.history['val_loss']

plt.figure(figsize=(15, 5))
plt.subplot(121)
# plt.subplot(121)은 1행 2열의 그리드에서 첫 번째 서브플롯

plt.plot(epoch, acc_train, 'b', marker='.', label='train_acc')
plt.plot(epoch, acc_valid, 'r--', marker='.', label='valid_acc')

plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epoch,loss_train,'b', marker='.', label='train_loss')
plt.plot(epoch,loss_valid,'r--', marker='.', label='valid_loss')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

