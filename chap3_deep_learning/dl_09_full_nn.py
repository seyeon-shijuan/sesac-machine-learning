import numpy as np

# weight, bias를 random으로 설정하여 순전파 역전파
class AffineFunction:
    def __init__(self, w1, w2, b):
        self.w1 = np.random.normal(0, 1, (1,))
        self.w2 = np.random.normal(0, 1, (1,))
        self.b = np.random.normal(0, 1, (1,))

    def forward(self, x1, x2):
        self.z = self.w1 * x1 + self.w2 * x2 + self.b
        return self.z


    def backward(self, dJ_dz):
        # 하는중
        pass


class Sigmoid:
    def forward(self, z):
        self.z = z
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self, dJ_dpred):
        # 0. 기본 식: 시그모이드 미분값(da_dz) * 로스 미분값(dJ_dpred)
        # 1. 시그모이드 미분값 da_dz = e^-z * (1+e^-z)**-2
        da_dz = np.exp(-self.z) * np.power((1 + np.exp(-self.z)), -2)
        # 2. 최종 식 da_dz * dJ_pred
        dJ_dz = da_dz * dJ_dpred
        return dJ_dz


class BCELoss:
    def forward(self, y, pred):
        self.pred = pred
        self.y = y
        J = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        return J

    def backward(self):
        # dJ /d yhat
        dJ_dpred = (self.pred - self.y) / (self.pred * (1 - self.pred))
        return dJ_dpred


class Model:
    def __init__(self):
        self.affine = AffineFunction()
        self.sigmoid = Sigmoid()
        self.loss = BCELoss()

    def forward(self, x1, x2, y):
        z = self.affine.forward(x1, x2)
        pred = self.sigmoid.forward(z)
        J = self.loss.forward(y, pred)
        return pred, J

    def backward(self):
        dJ_dpred = self.loss.backward()
        dJ_dz = self.sigmoid.backward(dJ_dpred)
        self.affine.backward(dJ_dz)



# AND
y = 0
model = Model()
pred, J = model.forward(1, 0, y)
model.backward()







