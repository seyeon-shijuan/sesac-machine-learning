import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def check_pattern_image_w_nptile():
    white_patch = 255 * np.ones(shape=(10, 10))
    black_patch = np.zeros(shape=(10, 10))

    img1 = np.hstack([white_patch, black_patch])
    img2 = np.hstack([black_patch, white_patch])
    img = np.vstack([img1, img2])

    tile = np.tile(img, reps=[2, 2])
    return tile


def visualize(names, if_vmax=False, *args):
    fig, axes = plt.subplots(ncols=3, figsize=(3 * len(args), 3))
    for i, data in enumerate(args):
        if not if_vmax:
            axes[i].imshow(data, cmap='gray')
        else:
            axes[i].imshow(data, cmap='gray', vmax=255, vmin=0)
        axes[i].set_title(names[i])
        axes[i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    fig.tight_layout()
    plt.show()


def get_data(data=None):
    if data is None:
        data = check_pattern_image_w_nptile()

    x_filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])  # 상하 대칭

    y_filter = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])  # 좌우 대칭

    return data, x_filter, y_filter


def two_dim_correlation(data, filter_):
    window_size = 3
    height, width = data.shape
    n_window_height = height - window_size + 1
    n_window_width = width - window_size + 1

    hadamard_product = lambda row, col: data[row:row + window_size, col:col + window_size] * filter_
    extracted = np.array(
        [[hadamard_product(row, col) for col in range(n_window_width)] for row in range(n_window_height)])
    correlated = np.sum(extracted, axis=(2, 3))

    return correlated


def sobel_filtering1():
    data, x_filter, y_filter = get_data()
    x_filtered = two_dim_correlation(data, x_filter)
    y_filtered = two_dim_correlation(data, y_filter)

    visualize(["data", "x_filtered", "y_filtered"], False, data, x_filtered, y_filtered)


def sobel_filtering2(path):
    img = Image.open(path)
    new_path = path.replace(".jpg", "-gray.jpg")
    img_gray = img.convert("L")
    if not os.path.isfile(new_path):
        img_gray.save(new_path)

    img_array = np.array(img_gray)
    data, x_filter, y_filter = get_data(img_array)
    x_filtered = two_dim_correlation(data, x_filter)
    y_filtered = two_dim_correlation(data, y_filter)

    visualize(["data", "x_filtered", "y_filtered"], True, data, x_filtered, y_filtered)


if __name__ == '__main__':
    # sobel_filtering1()
    sobel_filtering2(path="data/winter-3317660_640.jpg")
