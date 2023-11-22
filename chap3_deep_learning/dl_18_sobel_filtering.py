import numpy as np
import matplotlib.pyplot as plt

def np_ones_test():
    '''np.ones: 입력된 shape의 array를 만들고, 모두 1로 채워주는 함수'''

    tmp = np.ones(shape=(2, 3))
    # (2, 3)의 shape를 가지고 모두 1로 채워져있는 행렬을 만들어라
    print(tmp)


    '''ndarray에 scalar를 곱하면 원소별 곱셈'''
    tmp2 = 10 * tmp
    print(tmp2)


def check_pattern_image():
    white_patch = 255 * np.ones(shape=(10, 10))
    black_patch = 0 * np.ones(shape=(10, 10))

    img1 = np.hstack([white_patch, black_patch]) # [흰, 검] (10, 20)
    img2 = np.hstack([black_patch, white_patch]) # [검, 흰] (10, 20)
    img = np.vstack([img1, img2])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap='gray')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    plt.show()


def check_pattern_image2():
    white_patch = 255 * np.ones(shape=(10, 10))
    black_patch = np.zeros(shape=(10, 10))

    img1 = np.hstack([white_patch, black_patch, white_patch])
    img2 = np.hstack([black_patch, white_patch, black_patch])
    img = np.vstack([img1, img2, img1])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap='gray')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    plt.show()


def repeat_tile():
    data = np.arange(5)
    print(data)

    # np.repeat => 원소별 반복
    print("repeat:", np.repeat(data, repeats=3))

    # np.tile => 전체 반복
    print("tile:", np.tile(data, reps=3))


def repeat2():
    data = np.arange(6).reshape(2, 3)
    print(data)

    print(f"repeats=3, axis=0 \n {np.repeat(data, repeats=3, axis=0)}")
    print(f"repeats=3, axis=1 \n {np.repeat(data, repeats=3, axis=1)}")
    print(f"np.repeat(np.repeat(data, repeats=2, axis=0), repeats=3, axis=1) \n {np.repeat(np.repeat(data, repeats=2, axis=0), repeats=3, axis=1)}")


def tile2():
    data = np.arange(6).reshape(2, 3)
    print(data)

    print("tile(axis=0)")
    print(np.tile(data, reps=[3, 1]))
    print("tile(axis=1)")
    print(np.tile(data, reps=[1, 3]))
    print("tile(axis=0 and axis=1)")
    print(np.tile(data, reps=[3, 3]))


def check_pattern_image_w_nptile():
    white_patch = 255 * np.ones(shape=(10, 10))
    black_patch = np.zeros(shape=(10, 10))

    img1 = np.hstack([white_patch, black_patch])
    img2 = np.hstack([black_patch, white_patch])
    img = np.vstack([img1, img2])

    tile = np.tile(img, reps=[4, 4])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tile, cmap='gray')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()


def check_pattern_image_w_nptile2():
    white_patch = 255 * np.ones(shape=(10, 10))
    gray_patch = int(255/2) * np.ones(shape=(10, 10))
    black_patch = np.zeros(shape=(10, 10))

    img1 = np.hstack([white_patch, gray_patch])
    img2 = np.hstack([gray_patch, black_patch])
    img = np.vstack([img1, img2])

    tile = np.tile(img, reps=[4, 4])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tile, cmap='gray')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()


def grayscale_gradation_image_with_nprepeat():
    img = np.arange(0, 256, 50).reshape(1, -1)
    img = img.repeat(100, axis=0).repeat(30, axis=1)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow(img, cmap='gray')

    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()


def grayscale_gradation_image_with_nprepeat2():
    max_num = 151 # 256
    # n = int(max_num/4)
    img = np.arange(0, max_num, 50).reshape(1, -1)
    img = img.repeat(100, axis=0).repeat(30, axis=1)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow(img, cmap='gray', vmax=255, vmin=0)

    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()


def grayscale_gradation_image_with_nprepeat3():
    img = np.arange(0, 256, 50)[::-1].reshape(-1, 1)
    # img = np.arange(256, 0, -50).reshape(-1, 1)
    img = img.repeat(30, axis=0).repeat(100, axis=1)

    fig, ax = plt.subplots(figsize=(2, 4))
    ax.imshow(img, cmap='gray', vmax=256, vmin=0)

    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()


def grayscale_gradation_image_with_nprepeat4():
    img1 = np.arange(0, 256, 2).reshape(1, -1)
    img1 = img1.repeat(100, axis=0)
    img1 = img1.repeat(2, axis=1)

    img2 = np.arange(0, 256, 2)[::-1].reshape(1, -1)
    img2 = img2.repeat(100, axis=0).repeat(2, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(5, 5))
    axes[0].imshow(img1, cmap='gray', vmax=256, vmin=0)
    axes[1].imshow(img2, cmap='gray', vmax=256, vmin=0)

    axes[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axes[1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    fig.tight_layout()
    plt.show()


def one_dimensional_window_extraction():
    data = 10 * np.arange(1, 11)
    window_size = 3
    n_window = len(data) - window_size + 1
    print(f"{n_window=}")

    extracted = [data[i:i+window_size] for i in range(n_window)]
    print(extracted)


def two_dimensional_window_extraction():
    rows = np.arange(1, 6).reshape(-1, 1)
    cols = np.arange(7)
    data = (rows + cols) * 10

    window_size = 3
    height, width = data.shape
    n_window_height = height - window_size + 1
    n_window_width = width - window_size + 1
    print(f"{n_window_height=}, {n_window_width=}")

    extracted = np.array([[data[row:row+window_size, col:col+window_size] for col in range(n_window_width)] for row in range(n_window_height)])
    print(f"{extracted.shape=} => 3x3 커널이 3x5=15개")
    print(extracted)


if __name__ == '__main__':
    # check_pattern_image2()
    # repeat_tile()
    # repeat2()
    # tile2()
    # check_pattern_image_w_nptile()
    # check_pattern_image_w_nptile2()
    # grayscale_gradation_image_with_nprepeat()
    # grayscale_gradation_image_with_nprepeat2()
    # grayscale_gradation_image_with_nprepeat3()
    # grayscale_gradation_image_with_nprepeat4()
    # one_dimensional_window_extraction()
    two_dimensional_window_extraction()

