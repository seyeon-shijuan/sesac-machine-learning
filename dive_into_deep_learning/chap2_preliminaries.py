import torch
import os
import pandas as pd
import numpy as np
from d2l import torch as d2l
# conda install -c conda-forge d2l

def tensor_test():
    x = torch.arange(12, dtype=torch.float32)
    print(x)
    print(x.numel())
    print(x.shape)

    X = x.reshape(3, 4)
    print("X: ", X)

    print(torch.zeros((2, 3, 4)))
    print(torch.ones((2, 3, 4)))

    print(torch.randn(3, 4))
    print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

    print(X[-1])
    print(X[1:3])

    X[:2, :] = 12
    print(X)

    print(torch.exp(x))


def operations_test():
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y, x - y, x * y, x / y, x ** y)

    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(torch.cat((X, Y), dim=1))

    print(X == Y)
    print(X.sum())

    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a, b)
    print(a + b)
    print(a - b)
    print(a * b)

    before = id(Y)
    Y = Y + X
    print(id(Y) == before)

    Z = torch.zeros_like(Y)
    print('id(Z): ', id(Z))
    Z[:] = X + Y
    print('id(Z): ', id(Z))

    before = id(X)
    X += Y
    print(id(X) == before)


def conversion_to_other_python_objects():
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

    # conversion to other python objects
    A = X.numpy()
    B = torch.from_numpy(A)
    print(type(A), type(B))

    a = torch.tensor([3.5])
    print(a, a.item(), float(a), int(a))


def data_preprocessing_test():

    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('''NumRooms,RoofType,Price
    NA,NA,127500
    2,NA,106000
    4,Slate,178100
    NA,NA,140000''')

    data = pd.read_csv(data_file)
    print(data)

    inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)
    inputs = inputs.fillna(inputs.mean())
    print(inputs)

    X = torch.tensor(inputs.to_numpy(dtype=float))
    y = torch.tensor(targets.to_numpy(dtype=float))
    print(X, y)


def linear_algebra():
    # scalar
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    print(x+y, x*y, x/y, x**y)

    # vector
    x = torch.arange(3)
    print(x)

    print(x[2], len(x))
    print(x.shape)

    # matrices
    A = torch.arange(6).reshape(3, 2)
    print(A)
    print(A.T)

    A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print(A == A.T)

    # tensors
    print(torch.arange(24).reshape(2, 3, 4))

    # basic properties of tensor arithmetic
    A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    B = A.clone()
    print(A, A+B)
    print(id(A), id(A+B), (id(A) == id(A+B)))

    print(A * B)

    a = 2
    X = torch.arange(24).reshape(2, 3, 4)
    print(a + X, (a * X).shape)

    # reduction
    x = torch.arange(3, dtype=torch.float32)
    print(x, x.sum())

    A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print(A.shape, A.sum())
    tmp = A.numel()
    # numel = 개수(len)
    print(A.mean(), A.sum() / A.numel())
    print("shape:", A.shape) # shape[0]은 2
    print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

    # non reduction sum
    sum_A = A.sum(axis=1, keepdims=True)
    print(sum_A, sum_A.shape)

    print(A / sum_A)

    print("A:", A)
    # cumulative sum of elements
    print(A.cumsum(axis=0))
    # axis=0이면 각자 세로로 더하기

    # dot products
    y = torch.ones(3, dtype=torch.float32)
    print(x, y, torch.dot(x, y))

    print(torch.sum(x * y))

    # matrix-vector products
    print(f"A: {A} x: {x}")
    print(A.shape, x.shape, torch.mv(A, x), A@x)

    # 2x3 X 3x1 = 2x1

    # matrix-matrix multiplication
    # a_1^T는 첫번째 행벡터를 의미

    B = torch.ones(3, 4)
    print(torch.mm(A, B), A@B)

    # norms
    # l2 norm
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))
    # l1 norm
    print(torch.abs(u).sum())

    # Frobenius norm
    print(torch.norm(torch.ones(4, 9)))


def calculus_test():
    pass


if __name__ == '__main__':
    # tensor_test()
    # operations_test()
    # conversion_to_other_python_objects()
    # data_preprocessing_test()
    # linear_algebra()
    calculus_test()