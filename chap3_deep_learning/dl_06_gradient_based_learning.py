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
    def f(x): return 2 * x ** 2
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
        print(f"{iter + 1}-th Iteration")
        print(f"{dy_dx = :.4f}")
        print(f"next x: {x:.4f}\n")


# gbl2()

def gbl2_lr():
    def f(x): return 2 * x ** 2
    def df_dx(x): return 4 * x

    x = 3
    ITERATIONS = 3
    LR = 0.1
    x_track, y_track = [x], [f(x)]
    print(f"Initial x: {x}")

    for iter in range(ITERATIONS):
        dy_dx = df_dx(x)
        # x = x - dy_dx 기존
        x = x - LR * dy_dx
        x_track.append(x)
        y_track.append(f(x))
        print(f"{iter + 1}-th Iteration")
        print(f"{dy_dx = :.4f}")
        print(f"next x: {x:.4f}\n")

gbl2_lr()

def gradient_exploding():
    def f1(x): return 1/10 * x**2
    def df1_dx(x): return 1/5 * x

    def f2(x): return 1/5 * x**2
    def df2_dx(x): return 2/5 * x

    def f3(x): return 1/3 * x**2
    def df3_dx(x): return 2/3 * x

    x1, x2, x3 = 3, 3, 3
    ITERATIONS = 10
    x_track1, y_track1 = [x1], [f1(x1)]
    x_track2, y_track2 = [x2], [f2(x2)]
    x_track3, y_track3 = [x3], [f3(x3)]

    for iter in range(ITERATIONS):
        dy1_dx = df1_dx(x1)
        dy2_dx = df2_dx(x2)
        dy3_dx = df3_dx(x3)

        x1 = x1 - dy1_dx
        x2 = x2 - dy2_dx
        x3 = x3 - dy3_dx

        x_track1.append(x1)
        y_track1.append(f1(x1))
        x_track2.append(x2)
        y_track2.append(f2(x2))
        x_track3.append(x3)
        y_track3.append(f3(x3))

    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
    function_x = np.linspace(-5, 5, 100)

    axes[0].plot(function_x, f1(function_x), label='f1(x)', color='C0')
    axes[1].plot(function_x, f2(function_x), label='f1(x)', color='C1')
    axes[2].plot(function_x, f3(function_x), label='f1(x)', color='C2')

    axes[0].scatter(x_track1, y_track1, c=range(ITERATIONS+1), cmap='rainbow')
    axes[1].scatter(x_track2, y_track2, c=range(ITERATIONS+1), cmap='rainbow')
    axes[2].scatter(x_track3, y_track3, c=range(ITERATIONS+1), cmap='rainbow')

    for ax in axes: ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.show()

gradient_exploding()

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


# multivariate_case()