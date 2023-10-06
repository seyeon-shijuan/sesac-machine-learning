import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def dir_test():
    my_list = ['a', 'b', 'c']
    my_dict = {'a': 'apple', 'b': 'banana', 'c': 'carrot'}
    a = np.array([1, 2, 3])

    for attr in dir(a):
        print(attr)


def ndarray_test1():
    u = [1, 2, 3]
    v = [4, 5, 6]

    # numpy 없이 list로 원소별 덧셈하는 경우
    w = [a + b for a, b in zip(u, v)]
    print(w, type(w))

    # numpy로 원소별 덧셈하는 경우
    u = np.array(u)
    v = np.array(v)
    w = u + v
    print(w, type(w))


def ndarray_test2():
    scalar_np = np.array(3.14)
    vec_np = np.array([1, 2, 3])
    mat_np = np.array([[1, 2], [3, 4]])
    tensor_np = np.array([[[1, 2, 3],
                           [4, 5, 6]],

                          [[11, 12, 13],
                           [14, 15, 16]]]) # (2, 2, 3)

    print(scalar_np.shape) # ()
    print(vec_np.shape) # (3,)
    print(mat_np.shape) # (2, 2)
    print(tensor_np.shape) # (2, 2, 3)

    print('='*30)
    M = np.zeros(shape=(2, 3))

    print(M.shape)
    print(M)

    M = np.ones(shape=(2, 3))

    print(M.shape)
    print(M)

    M = np.full(shape=(2, 3), fill_value=3.14)
    print(M.shape)
    print(M)

    print('=' * 30)
    print(list(range(10)))
    print(list(range(2, 5)))
    print(list(range(2, 10, 2)))

    print(np.arange(10))
    print(np.arange(2, 5))
    print(np.arange(2, 10, 2))

    print(np.arange(10.5))
    print(np.arange(1.5, 10.5))
    print(np.arange(1.5, 10.5, 2.5))

    print('here')

    print(np.linspace(0, 1, 5))
    print(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(10, 5))
    random_values = np.random.randn(300)
    ax.hist(random_values, bins=20)
    print(random_values.shape)

    normal = np.random.normal(loc=[-2, 0, 3],
                              scale=[1, 2, 5],
                              size=(200, 3))
    print(normal.shape)

    normal = np.random.normal(loc=-2, scale=1, size=(3, 3))
    print(normal)

    # 표준정규분포(평균0 분산1) -> randn, uniform(0~1) -> rand

    uniform = np.random.rand(1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(uniform)
    print(uniform.shape)

    fig, ax = plt.subplots(figsize=(10, 5))
    uniform = np.random.uniform(low=-10, high=10, size=(10000, ))
    ax.hist(uniform, bins=20)

    randint = np.random.randint(low=0, high=7, size=(20,))
    print(randint)

    M = np.ones(shape=(10,))
    N = np.ones(shape=(3, 4))
    O = np.ones(shape=(3, 4, 5))
    P = np.ones(shape=(2, 3, 4, 5, 6))

    print("size of M: ", M.size)
    print("size of N: ", N.size)
    print("size of O: ", O.size)
    print("size of P: ", P.size)


def ndarray_test3():
    a = np.arange(6)
    b = np.reshape(a, (2, 3))

    print("original ndarray: \n", a)
    print("reshaped ndarray: \n", b)

    a = np.arange(24)
    b = np.reshape(a, (2, 3, 4))

    print("="*30)
    print("original ndarray: \n", a)
    print("reshaped ndarray: \n", b)

    a = np.arange(12)

    b = a.reshape((2, -1))
    c = a.reshape((3, -1))
    d = a.reshape((4, -1))
    e = a.reshape((6, -1))

    print(b.shape, c.shape, d.shape, e.shape)

    a = np.random.randint(0, 10, size=(2, 2))
    print(a)

    row_vector = a.reshape(1, -1)
    col_vector = a.reshape(-1, 1)
    print(row_vector.shape, col_vector.shape)

    print("="*30)
    # flatten : vector로 만들어 주는 것
    M = np.arange(9)
    N = M.reshape((3, 3))
    O = N.flatten()

    print(M, '\n')
    print(N, '\n')
    print(O, '\n')
    print(M.shape, O.shape)

    M = np.arange(27)
    N = M.reshape((3, 3, 3))
    O = N.flatten()
    print(O)

    a = np.random.randint(-5, 5, (5,))


def ndarray_test4():
    # 차원이 같은 경우
    A = np.arange(9).reshape(3, 3)
    B = 10 * np.arange(3).reshape((-1, 3))
    C = A + B

    A = np.arange(3).reshape((3, -1))
    B = 10 * np.arange(3).reshape((-1, 3))
    C = A + B

    print("A: {}/{}\n{}".format(A.ndim, A.shape, A))
    print("B: {}/{}\n{}\n".format(B.ndim, B.shape, B))

    print("A + B: {}/{}\n{}".format(A.ndim, C.shape, C))


    print("="*30)

    # (2,3,3) + (1,3,3)
    A = np.arange(2*3*3).reshape((2, 3, 3))
    B = 10*np.arange(3*3).reshape((1, 3, 3))

    C = A + B

    print("A: ", A)
    print("B: ", B)
    print("C: ", C)

    print('here')


def ndarray_test5():
    A = np.arange(2*3*4).reshape((2, 3, 4))
    B = 10*np.arange(3*4).reshape((3, 4))
    C = A + B

    print(C)


def ndarray_test6():
    a = np.arange(10)
    print(f"ndarray: \n{a}")
    # [0 1 2 3 4 5 6 7 8 9]
    print(f"a[:3]", a[:3])
    # a[:3] [0 1 2]
    print(f"a[::3]: ", a[::3])
    # a[::3]:  [0 3 6 9]

    a = np.arange(9).reshape((3, 3))
    print(f"ndarray: \n{a}")
    # [[0 1 2]
    #  [3 4 5]
    #  [6 7 8]]
    print(a[0][1])
    # 1

    a = np.arange(12).reshape((4, 3))
    print(f"ndarray: \n{a}")
    # [[ 0  1  2]
    #  [ 3  4  5]
    #  [ 6  7  8]
    #  [ 9 10 11]]
    print("a[1:3, 2]", a[1:3, 2])
    # a[1:3, 2] [5 8]


if __name__ == '__main__':
    # dir_test()
    # ndarray_test1()
    # ndarray_test2()
    # plt.show()
    # ndarray_test3()
    # ndarray_test4()
    # ndarray_test5()
    ndarray_test6()

