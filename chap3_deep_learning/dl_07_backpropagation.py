import numpy as np

# Chain Rule을 이용하여 미분계수 구하기

class Function1:
    def forward(self, x):
        z = x - 2
        return z

    def backward(self, dy_dz):
        dz_dx = 1
        dy_dx = dy_dz * dz_dx
        return dy_dx


class Function2:
    def forward(self, z):
        self.z = z
        y = 2 * (z**2)
        return y

    def backward(self):
        dy_dz = 4 * self.z
        return dy_dz


function1, function2 = Function1(), Function2()

x = 5
z = function1.forward(x)
print(f"{z=}")
y = function2.forward(z)
print(f"{y=}")

dy_dx = function1.backward(function2.backward())
print(f"{dy_dx=}")

print('==================')


class Function:
    def __init__(self):
        self.func1 = Function1()
        self.func2 = Function2()

    def forward(self, x):
        z = self.func1.forward(x)
        print(f"{z=}")
        y = self.func2.forward(z)
        print(f"{y=}")
        return y

    def backward(self):
        dy_dz = self.func2.backward()
        print(f"{dy_dz=}")
        dy_dx = self.func1.backward(dy_dz)
        print(f"{dy_dx=}")
        return dy_dx


fn = Function()
y = fn.forward(5)
dy_dx = fn.backward()
