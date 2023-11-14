import numpy as np

# Sigmoid 합성함수로 만들어서 순전파, 역전파

class Function1:
    def forward(self, z):
        self.z1 = -z
        return self.z1

    def backward(self, dy_dz1):
        dz1_dz = -1
        dy_dz = dz1_dz * dy_dz1
        return dy_dz

class Function2:
    def forward(self, z1):
        self.z2 = np.exp(z1)
        return self.z2

    def backward(self, dy_dz2):
        dz2_dz1 = self.z2 # e^x 를 미분하면 그대로
        dy_dz1 = dz2_dz1 * dy_dz2
        return dy_dz1


class Function3:
    def forward(self, z2):
        self.z3 = 1 + z2
        return self.z3

    def backward(self, da_dz3):
        dz3_dz2 = 1
        dy_dz2 = dz3_dz2 * da_dz3
        return dy_dz2



class Function4:
    def forward(self, z3):
        self.z3 = z3
        self.a = 1 / z3
        return self.a

    def backward(self):
        da_dz3 = - 1 / (self.z3 **2)
        return da_dz3


class Sigmoid:
    def __init__(self):
        self.fn1 = Function1()
        self.fn2 = Function2()
        self.fn3 = Function3()
        self.fn4 = Function4()

    def forward(self, z):
        z1 = self.fn1.forward(z)
        z2 = self.fn2.forward(z1)
        z3 = self.fn3.forward(z2)
        a = self.fn4.forward(z3)
        return a

    def backward(self):
        da_dz3 = self.fn4.backward()
        dy_dz2 = self.fn3.backward(da_dz3)
        dy_dz1 = self.fn2.backward(dy_dz2)
        dy_dz = self.fn1.backward(dy_dz1)
        return dy_dz


def sigmoid_test():
    sigmoid = Sigmoid()
    a = sigmoid.forward(0)
    print(f"{a=}")
    dy_dz = sigmoid.backward()
    print(f"{dy_dz=}")


