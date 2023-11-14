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


class Sigmoid:
    def forward(self, z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self, dJ_dpred):
        # da_dz



class BCELoss:
    def forward(self, y, pred):
        self.pred = pred
        self.y = y
        J = -(self.y * np.log(self.pred) + (1 - self.y) * np.log(1 - self.pred))
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
        self.sigmoid.backward(dJ_dpred)


# AND
y = 0
model = Model()
pred, J = model.forward(1, 0, y)
model.backward()







