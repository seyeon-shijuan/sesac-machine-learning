import numpy as np

# Chain Rule을 이용하여 미분계수 구하기

class Function1:
    def forward(self, x):
        z = x - 2
        return z

    def backward(self, dy_dz):
        dz_dx = 1
        dy_dx = dy_dz * dz_dx
        return dy_dz


class Function2:
    def forward(self, z):
        self.z = z
        y = 2 * (z**2)
        return y

    def backward(self):
        dy_dz = 4 * self.z
        return dy_dz

# 만들고 DM 보내기

class AND:
    def __init__(self):
        self.w1 = 0.5
        self.w2 = 0.5
        self.b = -0.7

    def forward(self, x1, x2):
        z = self.w1 * x1 + self.w2 * x2 + self.b
        return z

    def backward(self, dy_dz):
        # 아직 안했음
        dz_dx = 1
        dy_dx = dy_dz * dz_dx
        return dy_dz


class ActivationFunction:
    def forward(self, z):
        out = 1 / 1 + np.exp(-z)
        return out

    def backward(self, dy_dz):
        dz_dx = 1
        dy_dx = dy_dz * dz_dx
        return dy_dz


class Function:
    def __init__(self):
        self.function1 = AND()
        self.function2 = ActivationFunction()

    def forward(self, x_val):
        z = self.function1.forward(*x_val)
        print(f"{z=}")
        out = self.function2.forward(z)
        print(f"{out=}")


and_gate = Function()
and_gate.forward((1, 1))



