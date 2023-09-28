import torch
import os
import pandas as pd


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



if __name__ == '__main__':
    # tensor_test()
    # operations_test()
    # conversion_to_other_python_objects()
    # data_preprocessing_test()
    linear_algebra()
