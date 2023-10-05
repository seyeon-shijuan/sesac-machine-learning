import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

from sklearn import datasets
from sklearn.model_selection import train_test_split


def perceptron_test():
    input_data = np.array([[2, 3], [5, 1]])
    print(input_data)
    x = input_data.reshape(-1)
    print(x)

    # 가중치 및 편향
    w1 = np.array([2, 1, -3, 3])
    w2 = np.array([1, -3, 1, 3])
    b1 = 3
    b2 = 3

    # 가중 합
    W = np.array([w1, w2])
    print(W)
    b = np.array([b1, b2])
    print(b)
    weight_sum = np.dot(W, x) + b
    print(weight_sum)

    # 출력층
    res = 1 / (1+np.exp(-weight_sum))
    print(res)


def tensorflow_test1():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(32, 32, 1)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()

    model.save('cnn_model.h5')

    cnn_model2 = load_model('cnn_model.h5')


def tensorflow_test2():
    input_layer = Input(shape=(32, 32, 1))
    x = Dense(units=100, activation='relu')(input_layer)
    x = Dense(units=50, activation='relu')(x)
    output_layer = Dense(units=5, activation='softmax')(x)
    model2 = Model(input_layer, output_layer)
    model2.summary()


def classify_wine():
    np.random.seed(0)
    tf.random.set_seed(0)

    raw_wine = datasets.load_wine()
    X = raw_wine.data
    y = raw_wine.target
    print(X.shape)
    print(np.unique(y), set(y))

    y_hot = to_categorical(y)
    # print(y_hot)

    X_tn, X_te, y_tn, y_te = train_test_split(X, y_hot, random_state=0)

    n_feat = X_tn.shape[1] # 133
    n_class = len(set(y))
    epo = 30

    model = Sequential()
    model.add(Dense(20, input_dim=n_feat))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_tn, y_tn, epochs=epo, batch_size=5)
    tmp = model.evaluate(X_tn, y_tn) # loss, accuracy
    print(model.evaluate(X_tn, y_tn)[1]) # accuracy
    print(model.evaluate(X_te, y_te)[1]) # accuracy

    # 정확도 및 손실 정도 시각화 준비
    epoch = np.arange(1, epo+1)
    accuracy = hist.history['accuracy']
    loss = hist.history['loss']

    # 그래프
    plt.plot(epoch, accuracy, label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.plot(epoch, loss, 'r', label='loss')
    # r : 플랏 색깔이 빨간색
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def regression_boston():
    np.random.seed(0)
    tf.random.set_seed(0)

    raw_boston = datasets.load_boston()
    X = raw_boston.data
    y = raw_boston.target

    print(X.shape)
    print(set(y))
    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=0)

    n_feat = X_tn.shape[1]
    epo = 30

    model = Sequential()
    model.add(Dense(20, input_dim=n_feat, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    hist = model.fit(X_tn, y_tn, epochs=epo, batch_size=5)
    print(model.evaluate(X_tn, y_tn)[1])
    print(model.evaluate(X_te, y_te)[1])

    # 시각화
    epoch = np.arange(1, epo+1)
    mse = hist.history['mean_squared_error']
    loss = hist.history['loss']

    # fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # axes.flat[0].plot(epoch, mse, label='mse')
    # axes.flat[0].set_xlabel('epoch')
    # axes.flat[0].set_ylabel('mean_squared_error')
    # axes.flat[0].legend()
    #
    # axes.flat[1].plot(epoch, loss, 'r', label='loss')
    # axes.flat[1].set_xlabel('epoch')
    # axes.flat[1].set_ylabel('loss')
    # axes.flat[1].legend()

    # plt.show()

    pred_y = model.predict(X_te).flatten()
    res_df = pd.DataFrame(pred_y, columns=['predict_val'])
    res_df['real_val'] = y_te
    df_sort = res_df.sort_values(["predict_val"], ascending=True)

    idx = np.arange(1,len(df_sort)+1)
    plt.scatter(idx, df_sort['real_val'], marker='o', label='real_val')
    plt.plot(idx, df_sort['predict_val'], color='r', label='predict_val')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # perceptron_test()
    # tensorflow_test1()
    # tensorflow_test2()
    # classify_wine()
    regression_boston()