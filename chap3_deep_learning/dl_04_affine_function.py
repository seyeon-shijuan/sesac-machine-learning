import numpy as np
import matplotlib.pyplot as plt


class AffineFunction:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        affine = np.dot(self.w, x) + self.b
        return affine


class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-z))


class ArtificialNeuron:
    def __init__(self, w, b):
        self.affine = AffineFunction(w, b)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        z = self.affine.forward(x)
        output = self.sigmoid.forward(z)
        return output


class Model:
    def __init__(self):
        self.and_gate = ArtificialNeuron(w=np.array([0.5, 0.5]), b=-0.7)
        self.or_gate = ArtificialNeuron(w=np.array([0.5, 0.5]), b=-0.2)
        self.nand_gate = ArtificialNeuron(w=np.array([-0.5, -0.5]), b=0.7)

    def forward(self, x: np.ndarray):
        a1 = self.and_gate.forward(x)
        a2 = self.or_gate.forward(x)
        a3 = self.nand_gate.forward(x)

        return np.array([a1, a2, a3])


affine1 = AffineFunction(np.array([1, 1]), -1.5)
print(f"{affine1.w=}, {affine1.b=}")
vec1 = np.array([0, 1])
print(f"{affine1.forward(vec1)=}")

affine2 = AffineFunction(np.array([-1, -1]), 0.5)
print(f"{affine2.w=}, {affine2.b=}")
vec2 = np.array([0, 0])
print(f"{affine2.forward(vec2)=}")

sigmoid = Sigmoid()
print(f"{sigmoid.forward(-5)=}")
print(f"{sigmoid.forward(-3)=}")
print(f"{sigmoid.forward(0)=}")
print(f"{sigmoid.forward(3)=}")
print(f"{sigmoid.forward(5)=}")


fig, ax = plt.subplots(figsize=(6, 2.5))

x_data = np.linspace(-5, 5, 100)
y_data = sigmoid.forward(x_data)

# line chart
ax.plot(x_data, y_data)

# vertical & horizontal lines
ax.axvline(x=0, ymin=0, ymax=1, color='gray', linewidth=1)
ax.axhline(y=0, xmin=0, xmax=1, color='gray', linewidth=1)

ax.set_title("Sigmoid Function")
plt.show()

an = ArtificialNeuron(np.array([0.5, 0.5]), -0.7)
print(f"{an.forward(np.array([1, 1]))=}")

model = Model()
print(f"{model.forward(np.array([0, 1]))=}")