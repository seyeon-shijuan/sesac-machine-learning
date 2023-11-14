import numpy as np

# weight, bias를 random으로 설정하여 순전파 역전파
class AffineFunction:
    def __init__(self):
        self.w = np.random.randn(2)
        self.b = np.random.randn()

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

        # 2. 파라미터 업데이트
        self.w -= lr * dz_dw * dJ_dz
        self.b -= lr * dz_db * dJ_dz
        return self.w, self.b



class Sigmoid:
    def forward(self, z):
        self.z = z
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self, dJ_dpred):
        # 0. 기본 식: 시그모이드 미분값(da_dz) * 로스 미분값(dJ_dpred)

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
        self.affine = AffineFunction()
        self.sigmoid = Sigmoid()

    def forward(self, X):
        z = self.affine.forward(X)
        pred = self.sigmoid.forward(z)
        return pred

    def backward(self, dJ_dpred, lr):
        dJ_dz = self.sigmoid.backward(dJ_dpred)
        w, b = self.affine.backward(dJ_dz, lr)
        return w, b

def train():
    # AND
    X = np.array([1, 0])
    y = 0
    model = Model()
    loss_fn = BCELoss()
    lr = 0.1

    print(f"AND GATE Train with {X=} {y=}")

    for i in range(300):
        pred = model.forward(X)
        loss, dJ_dpred = loss_fn(pred, y)
        print(f"{X=}, {y=}, {pred=:.4f}, {loss=:.4f}")
        w, b = model.backward(dJ_dpred, lr)
        print(f"{w=}, {b=:.4f}", end=" /\t")


def train_andgate():
    # AND
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_data = [0, 0, 0, 1]
    model = Model()
    loss_fn = BCELoss()
    lr = 0.1

    # print(f"AND GATE Train with {X=} {y=}")

    for i in range(2000):
        X = X_data[i % 4]
        y = y_data[i % 4]
        pred = model.forward(X)
        loss, dJ_dpred = loss_fn(pred, y)
        print(f"{X=}, {y=}, {pred=:.4f}, {loss=:.4f}")
        w, b= model.backward(dJ_dpred, lr)
        print(f"{w=}, {b=:.4f}", end=" /\t")


if __name__ == '__main__':
    # train()
    train_andgate()






