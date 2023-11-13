import matplotlib.pyplot as plt
import numpy as np


def gbl1():
    def f(x):
        return 1 / 10 * x ** 2

    def df_dx(x):
        return 1 / 5 * x

    x = 3
    ITERATIONS = 20
    x_track = list()
    x_track.append(x)
    y_track = list()
    y_track.append(f(x))
    print(f"Initial x: {x}")

    for iter in range(ITERATIONS):
        dy_dx = df_dx(x)
        x = x - dy_dx
        x_track.append(x)
        y_track.append(f(x))
        print(f"{iter + 1}-th x: {x:.4f}")

    function_x = np.linspace(-5, 5, 100)
    function_y = f(function_x)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    axes[0].plot(function_x, function_y)
    axes[0].scatter(x_track, y_track, c=range(ITERATIONS + 1), cmap='rainbow')
    axes[0].set_xlabel('x', fontsize=15)
    axes[0].set_ylabel('y', fontsize=15)

    axes[1].plot(x_track, marker='o')
    axes[1].set_xlabel('iteration', fontsize=15)
    axes[1].set_ylabel('x', fontsize=15)
    # fig.tight_layout()
    # plt.show()


def gbl2():
    def f(x): return 2 * x**2
    def df_dx(x): return 4 * x

    x = 3
    ITERATIONS = 3
    x_track, y_track = [x], [f(x)]
    print(f"Initial x: {x}")

    for iter in range(ITERATIONS):
        dy_dx = df_dx(x)
        x = x - dy_dx
        x_track.append(x)
        y_track.append(f(x))
        print(f"{iter + 1}-th x: {x:.4f}")
        print(f"{dy_dx = :.4f}")
        print(f"next x: {x:.4f} \n")


def gradient_exploding():
    def f1(x): return 1/10 * x**2
    # def df1_dx(x):


# gradient_exploding()

def multivariate_case():
    # multivariate case
    def f(x1, x2): return x1**2 + x2**2


    def df_dx(x): return 2*x


    x1, x2 = 3, -2
    ITERATIONS = 10
    LR = 0.1
    x1_track, x2_track = [x1], [x2]
    y_track = [f(x1, x2)]

    print(f"Initial (x1, x2): ({x1}, {x2})")
    print(f"Initial y: {f(x1, x2)}")

    for iter in range(ITERATIONS):
        dy_dx1 = df_dx(x1)
        dy_dx2 = df_dx(x2)

        x1 = x1 - LR * dy_dx1
        x2 = x2 - LR * dy_dx2

        x1_track.append(x1); x2_track.append(x2); y_track.append(f(x1, x2))
        print(f"{iter +1}-th (x1, x2): ({x1:.3f}, {x2:.3f})")
        print(f"y: {f(x1, x2):.3f}")


    # Bivariate Function을 Contour Plot으로 시각화 하기
    def f(x1, x2): return x1**2 + x2**2

    function_x1 = np.linspace(-5, 5, 100)
    function_x2 = np.linspace(-5, 5, 100)

    function_X1, function_X2 = np.meshgrid(function_x1, function_x2)
    function_Y = np.log(f(function_X1, function_X2))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(function_X1, function_X2, function_Y, levels=100, cmap='Reds_r')
    ax.plot(x1_track, x2_track, marker='o', label='Optimization Path')

    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.show()


multivariate_case()