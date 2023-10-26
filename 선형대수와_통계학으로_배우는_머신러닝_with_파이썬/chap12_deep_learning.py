import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence


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


def mnist_test():
    np.random.seed(0)
    tf.random.set_seed(0)
    (X_tn0, y_tn0), (X_te0, y_te0) = datasets.mnist.load_data()

    print(X_tn0.shape)
    print(y_tn0.shape)
    print(X_te0.shape)
    print(y_te0.shape)

    plt.figure(figsize=(10, 5))
    for i in range(2*5):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_tn0[i].reshape((28, 28)), cmap='Greys')

    # plt.show()

    X_tn_re = X_tn0.reshape(60000, 28, 28, 1)
    X_tn = X_tn_re / 255
    print(X_tn.shape)

    X_te_re = X_te0.reshape(10000, 28, 28, 1)
    X_te = X_te_re / 255
    print(X_te.shape)


def cnn():
    np.random.seed(0)
    tf.random.set_seed(0)

    (X_tn0, y_tn0), (X_te0, y_te0) = datasets.mnist.load_data()

    def show_img():
        plt.figure(figsize=(10, 5))
        for i in range(2*5):
            plt.subplot(2, 5, i+1)
            plt.imshow(X_tn0[i].reshape((28, 28)), cmap='Greys')

        plt.show()


    print(set(y_tn0))
    print(f"{X_tn0.shape}")
    X_tn_re = X_tn0.reshape(60000, 28, 28, 1)
    X_tn = X_tn_re / 255
    X_te_re = X_te0.reshape(10000, 28, 28, 1)
    X_te = X_te_re / 255
    print(X_te.shape)

    # y라벨이 10개 종류니까 원핫인코딩하면 가로축이 10개가 됨
    y_tn = to_categorical(y_tn0)
    y_te = to_categorical(y_te0)

    # 합성곱 신경망 생성

    n_class = len(set(y_tn0))
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     input_shape=(28, 28, 1),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    model.summary()

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 학습
    hist = model.fit(X_tn, y_tn, epochs=3, batch_size=100)

    # 모형 평가
    print(f"model.evaluate training = {model.evaluate(X_tn, y_tn)[1]}")
    print(f"model.evaluate test = {model.evaluate(X_te, y_te)}")

    # 오답 데이터 확인
    y_pred_hot = model.predict(X_te)
    print(y_pred_hot[0])
    y_pred = np.argmax(y_pred_hot, axis=1)
    print(y_pred)
    diff = y_te0 - y_pred
    diff_idx = []
    y_len = len(y_te0)
    for i in range(0, y_len):
        if(diff[i] !=0):
            diff_idx.append(i)

    # 오답 데이터 시각화
    plt.figure(figsize=(10, 5))
    for i in range(2*5):
        plt.subplot(2, 5, i+1)
        raw_idx = diff_idx[i]
        plt.imshow(X_te0[raw_idx].reshape(28, 28), cmap='Greys')

    plt.show()


(X_tn0, y_tn0), (X_te0, y_te0) = datasets.imdb.load_data(num_words=2000)
print(X_tn0.shape)
print(y_tn0.shape)
print(X_te0.shape)
print(y_tn0.shape)

X_train = X_tn0[0:20000]
y_train = y_tn0[0:20000]
X_valid = X_tn0[20000:25000]
y_valid = y_tn0[20000:25000]

print(X_train[0])

# 개별 피처
print(f"len(X_train[0]) = {len(X_train[0])}")
print(f"len(X_train[0]) = {len(X_train[1])}")

# 타깃 클래스 확인
print(f"set(y_te0) = {set(y_te0)}")
print(f"len(set(y_te0)) = {len(set(y_te0))}")

# 피처 데이터 변형
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_valid = sequence.pad_sequences(X_valid, maxlen=100)
X_test = sequence.pad_sequences(X_te0, maxlen=100)
print(f"X_train, X_valid, X_test = {X_train.shape, X_valid.shape, X_test.shape}")


# if __name__ == '__main__':
#     # perceptron_test()
#     # tensorflow_test1()
#     # tensorflow_test2()
#     # classify_wine()
#     # regression_boston()
#     # cnn()