import numpy as np


def one_dim_correlation():
    print('for loop ver')
    filter = np.array([-1, 1, -1])
    data = np.array([-1, 0, -1, 0, 0, 1, -1, 1, -1, -1])

    window_size = 3
    n_window = len(data) - window_size +1
    correlated = np.array([np.dot(data[i:i+window_size], filter) for i in range(n_window)])
    print(f"{correlated=}")


def two_dim_correlation():
    rows = np.arange(1, 6).reshape(-1, 1)
    cols = np.arange(7)
    data = (rows + cols) * 10

    window_size = 3
    height, width = data.shape
    n_window_height = height - window_size + 1
    n_window_width = width - window_size + 1

    filter = np.array([
        [1, 2, 5],
        [-10, 2, -2],
        [5, 1, -4]
    ])

    # data에서 window로 추출한 (3,3)과 filter (3,3)을 원소곱(hadamard product)
    # 2차원부터는 np.dot()을 하면 행렬곱을 계산하기 때문에 사용할 수 없고
    # * 연산으로 hadamard product한 후 np.sum()을 해야 함
    hadamard_product = lambda row, col: data[row:row + window_size, col:col + window_size] * filter
    extracted = np.array([[hadamard_product(row, col) for col in range(n_window_width)] for row in range(n_window_height)])
    # extracted = (3, 5, 3, 3) -> (3, 3)인 원소곱한 결괏값이 3x5=15개
    print("after hadamard")
    print(extracted[0, 0])

    correlated = np.sum(extracted, axis=(2, 3))
    print(f"correlated = {correlated.shape} \n{correlated}")


def one_dim_corr_np1():
    # 내가 작성한 알고리즘
    print("numpy broadcasting ver. 1")
    np.random.seed(0)
    data = np.random.randint(-1, 2, (10,))
    filter_ = np.array([-1, 1, -1])
    print(f"{data=}")
    print(f"{filter_=}")
    L = len(data)
    F = len(filter_)

    L_ = L - F + 1
    filter_idx = np.tile(np.arange(F), reps=[L_, 1])
    window_idx = np.arange(L_).reshape(-1, 1)
    idx_arr = filter_idx + window_idx
    print("idx_arr")
    print(idx_arr)
    sliding_window = data[idx_arr]
    hadamard = sliding_window * filter_
    summation = np.sum(hadamard, axis=1)
    print(f"{summation=}")


def one_dim_corr_np2():
    # 교안 코드
    np.random.seed(0)
    data = np.random.randint(-1, 2, (10,))
    filter_ = np.array([-1, 1, -1])
    print(f"{data=}")
    print(f"{filter_=}")
    L = len(data)
    F = len(filter_)
    L_ = L - F + 1

    filter_idx = np.arange(F).reshape(1, -1)
    window_idx = np.arange(L_).reshape(-1, 1)
    idx_arr = filter_idx + window_idx
    print("idx_arr")
    print(idx_arr)
    window_mat = data[idx_arr]
    multiplied = np.matmul(window_mat, filter_)
    print(f"{multiplied=}")


if __name__ == '__main__':
    # one_dim_correlation()
    # two_dim_correlation()
    # one_dim_corr_np1()
    one_dim_corr_np2()
