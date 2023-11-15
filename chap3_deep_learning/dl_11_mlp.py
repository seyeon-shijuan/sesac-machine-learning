import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class AffineFunction:
    def __init__(self, in_size, out_size):
        self.w = np.random.randn(in_size, out_size)
        self.b = np.random.randn(out_size)

    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.w) + self.b
        return self.z

    def backward(self, dJ_dz, lr):
        # 0. 기본 식:
        # w1 := w1 - LR * [w1에 대한 미분값(dz_dw1) * 시그모이드에서 넘어온 미분값(dJ_dz)]
        # w2 := w2 - LR * [w2에 대한 미분값(dz_dw2) * 시그모이드에서 넘어온 미분값(dJ_dz)]
        # b := b - LR * [b에 대한 미분값(dz_db) * 시그모이드에서 넘어온 미분값(dJ_dz)]

        # 1. 파라미터 별 미분 값 구하기
        dz_dw = self.X
        dz_db = 1

        curr_dim = 1 if np.isscalar(dJ_dz) else len(dJ_dz)

        # 2. 파라미터 업데이트
        dJ_dw = (dz_dw * dJ_dz).reshape(-1, curr_dim)
        dJ_db = dz_db * dJ_dz

        self.w -= lr * dJ_dw
        self.b -= lr * dJ_db

        # self.w -= lr * dz_dw * dJ_dz
        # self.b -= lr * dz_db * dJ_dz
        # return self.w, self.b
        return self.w



class Sigmoid:
    def forward(self, z):
        # z가 scalar인 경우 array []를 풀기
        if z.shape[0] == 1:
            self.z = z[0]
            self.a = 1 / (1 + np.exp(-self.z))
            return self.a

        self.z = z
        self.a = 1 / (1 + np.exp(-self.z))
        return self.a


    def backward(self, dJ_dpred):
        # 0. 기본 식: 시그모이드 미분값(da_dz) * 로스 미분값(dJ_dpred)

        # 2개 이상의 노드에서 sigmoid backpropagation을 하는 경우
        if not np.isscalar(dJ_dpred):
            dJ_dpred = dJ_dpred.flatten()

        # 1. 시그모이드 미분값 da_dz = a(1-a)
        da_dz = self.a * (1 - self.a)
        # 2. 최종 식 da_dz * dJ_pred
        dJ_dz = da_dz * dJ_dpred
        return dJ_dz



class BCELoss:
    def __call__(self, pred, y):
        # 1. loss 계산
        J = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        # 2. loss 미분 값계산:  dJ /d yhat
        dJ_dpred = (pred - y) / (pred + 1e-7 * (1 - pred + 1e-7))
        # 분모의 + 1e-7는 ZeroDivision Error 막기 위해 추가함 (0으로 나누면 에러나서)
        return J, dJ_dpred


class Model:
    def __init__(self):
        self.affine1 = AffineFunction(2, 2)
        self.sigmoid1 = Sigmoid()
        self.affine2 = AffineFunction(2, 1)
        self.sigmoid2 = Sigmoid()

    def forward(self, X):
        z1 = self.affine1.forward(X)
        a1 = self.sigmoid1.forward(z1)
        z2 = self.affine2.forward(a1)
        pred = self.sigmoid2.forward(z2)
        return pred

    def backward(self, dJ_da2, lr):
        dJ_dz2 = self.sigmoid2.backward(dJ_da2)
        dJ_da1 = self.affine2.backward(dJ_dz2, lr)

        dJ_dz1 = self.sigmoid1.backward(dJ_da1)
        dJ_dX = self.affine1.backward(dJ_dz1, lr)

        return dJ_dX



def calculate_accuracy(preds, targets):
    rounded_preds = np.round(preds)
    correct = (rounded_preds == targets).sum().item()
    accuracy = correct / len(targets)
    return accuracy


def main_routine():
    np.random.seed(8000) # 3500 3506 8000
    N_SAMPLES = 100
    LR = 0.001
    EPOCHS = 30 # 30

    X, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=2,
                      cluster_std=0.5, random_state=0)

    '''Instantiation'''
    model = Model()
    loss_fn = BCELoss()

    epoch_accuracy = list()
    epoch_loss = list()

    for epoch in range(EPOCHS):
        pred_list = list()
        loss_list = list()

        for X_, y_ in zip(X, y):
            '''Training'''
            pred = model.forward(X_)
            loss, dJ_dpred = loss_fn(pred, y_)
            model.backward(dJ_dpred, LR)

            '''Metric(loss, accuracy) Calculations'''
            pred_list.append(pred)
            loss_list.append(loss)

        epoch_accuracy.append(calculate_accuracy(pred_list, y))
        epoch_loss.append(sum(loss_list)/len(loss_list))

    '''Result Visualization'''
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(epoch_loss, linestyle='-')
    axes[0].set_ylabel("BCELoss", fontsize=15)
    axes[1].plot(epoch_accuracy, linestyle='-')
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[1].set_xlabel("Epoch", fontsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_routine()