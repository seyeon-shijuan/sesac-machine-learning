import matplotlib.pyplot as plt
import numpy as np


def fig_test1():
    # subplots
    # fig = plt.figure()
    # fig = plt.figure(figsize=(7, 7))
    # fig = plt.figure(figsize=(7, 7), facecolor='linen')

    ax = plt.subplot()


def fig_test2():
    # title
    figsize = (7, 7)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("title of a figure", fontsize=30, fontfamily='monospace')
    ax.set_title("title of an ax", fontsize=20, fontfamily='monospace')

    ax.set_xlabel("x label", fontsize=20, color='darkblue', alpha=0.7)
    ax.set_ylabel("y label", fontsize=20, color='darkblue', alpha=0.5)

    fig.tight_layout()


def fig_test3():
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.set_xlim([0, 100])

    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    ax2.set_ylim([0, 0.1])

    ax1.set_title("twinx graph", fontsize=30)
    ax1.set_ylabel("data1", fontsize=20)
    ax2.set_ylabel("data2", fontsize=20)

    fig.tight_layout()


def fig_test4():
    # ticks
    fig, ax = plt.subplots(figsize=(7, 7))
    # 틱의 숫자 크기
    # ax.tick_params(labelsize=20, length=10, width=3, bottom=False, labelbottom=False)
    # ax.tick_params(labelsize=20, length=10, width=3, left=False, labelleft=False)
    ax.tick_params(axis='x',
                   labelsize=20, length=10, width=3, bottom=False, labelbottom=False,
                   top=True, labeltop=True,
                   left=False, labelleft=False,
                   right=True, labelright=True,
                   rotation=-30)


def fig_test5():
    # ax text
    figsize = (7, 7)
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    ax.grid()

    ax.tick_params(axis='both', labelsize=15)
    # ax.text(x=0, y=0, s='hello', fontsize=30)

    ax.tick_params(axis='both', labelsize=15)

    # ax.text(x=0, y=0, s="hello1", fontsize=30)
    # ax.text(x=0.5, y=0, s="hello2", fontsize=30)
    # ax.text(x=0.5, y=-0.5, s="hello3", fontsize=30)

    # horizontal alignment: center, left, right
    # vertical alignment: center, top, bottm

    ax.text(x=0, y=0, va='top', ha='left', s="hello", fontsize=30)
    ax.text(x=0, y=0, va='center', ha='center', s="hello", fontsize=30)
    ax.text(x=0, y=0, va='bottom', ha='right', s='hello', fontsize=30)


def fig_test6():
    figsize = (7, 7)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0, 10])
    # ax.set_xticks([0, 1, 5, 10])
    xticks = [i for i in range(10)]
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=20)


def color_test():
    color_list = ['b', 'g', 'r', 'c', 'm', 'y']

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, len(color_list)])

    for c_idx, c in enumerate(color_list):
        ax.text(0, c_idx, "color="+c, fontsize=20, ha='center', color=c)


def scatter_plot():
    np.random.seed(0)

    n_data = 500
    # np.random.normal= 정규분포에서 랜덤 선택 loc 평균 scale 표준 편차
    x_data = np.random.normal(0, 1, size=(n_data,))
    y_data = np.random.normal(0, 1, size=(n_data,))

    # 전 구간에서 동일한 확률로 랜덤 선택 low 최솟값 high 최댓값
    s_arr = np.random.uniform(100, 500, n_data)
    c_arr = [np.random.uniform(0, 1, 3) for _ in range(n_data)]

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(x_data, y_data, s=s_arr, c=c_arr, alpha=0.3)
    ax.scatter(x_data, y_data, s=s_arr, c=c_arr, alpha=0.3, linewidth=5)


def scatter_plot2():
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.scatter(0, 0, s=10000, facecolor='red', edgecolor='b', linewidth=5)

    n_data = 100
    x_data = np.random.normal(0, 1, (n_data,))
    y_data = np.random.normal(0, 1, (n_data,))

    fig, ax = plt.subplots(figsize=(5, 5))
    # ax.scatter(x_data, y_data,  s=300, facecolor='white', edgecolor='tab:blue',
    #            linewidth=5)
    # s = 점 사이즈
    # ax.scatter(x_data, y_data, s=50, facecolor="None", edgecolor="tab:blue",
    #            linewidth=5)

    ax.scatter(x_data, y_data, s=300, facecolor='None', edgecolor='tab:blue',
               linewidth=5, alpha=0.5)


def time_series():
    n_data = 100
    s_idx = 30
    # 1번부터 2번까지 1씩 증가하면서 array
    x_data = np.arange(s_idx, s_idx + n_data)
    y_data = np.random.normal(0, 1, (n_data, ))
    print(x_data)
    print(y_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_data, y_data)

    fig.tight_layout(pad=3)
    x_ticks = np.arange(s_idx, s_idx + n_data + 1, 20)
    ax.set_xticks(x_ticks)
    ax.tick_params(labelsize=25)
    ax.grid()


def time_series2():
    np.random.seed(0)

    x_data = np.array([10, 25, 31, 40, 55, 80, 100])
    y_data = np.random.normal(0, 1, (7, ))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_data, y_data)

    fig.subplots_adjust(left=0.2)
    ax.tick_params(labelsize=25)

    ax.set_xticks(x_data)
    # 최솟값 최댓값을 tuple로 가져오기
    ylim = ax.get_ylim()
    yticks = np.linspace(ylim[0], ylim[1], 8)
    ax.set_yticks(yticks)

    ax.grid()


def sin_graph():
    PI = np.pi
    t = np.linspace(-4*PI, 4*PI, 300)
    sin = np.sin(t)
    linear = 0.1*t

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, sin)
    ax.plot(t, linear)

    ax.set_ylim([-1.5, 1.5])

    x_ticks = np.arange(-4*PI, 4*PI+0.1, PI)
    x_ticklabels = [str(i) + r'$\pi$' for i in range(-4, 5)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    ax.tick_params(labelsize=20)
    ax.grid()


def trigonometric_functions():
    PI = np.pi
    t = np.linspace(-4*PI, 4*PI, 1000)
    sin = np.sin(t)
    cos = np.cos(t)
    tan = np.tan(t)
    tan[:-1][np.diff(tan) < 0] = np.nan

    # 4개가 되면 boolean indexing이 가능해져서 not a number 처리 가능
    # y[:-1][np.diff(y) < 0] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(7, 10))

    axes[0].plot(t, sin)
    axes[1].plot(t, cos)
    axes[2].plot(t, tan)
    # tanh 이상함

    fig.tight_layout()
    axes[2].set_ylim([-5, 5])


def trigonometric_functions2():
    PI = np.pi
    t_ = np.linspace(-4*PI, 4*PI, 1000)
    t = np.linspace(-4*PI, 4*PI, 1000).reshape(1, -1)
    print(f"t_ shape :{t_.shape}, t shape : {t.shape}")
    # t_ shape: (1000,) vector라는 의미 , t shape: (1, 1000) matrix라는 의미

    sin = np.sin(t)
    cos = np.cos(t)
    tan = np.tan(t)
    data = np.vstack((sin, cos, tan))
    # data = np.hstack((sin, cos, tan))
    print(data.shape)

    title_list = [r'$sin(t)$', r'$cos(t)$', r'$tan(t)$']
    x_ticks = np.arange(-4*PI, 4*PI+PI, PI)
    x_ticklabels = [str(i) + r'$\pi$' for i in range(-4, 5)]

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    tmp = axes.flat

    for ax_idx, ax in enumerate(axes.flat):
        ax.plot(t.flatten(), data[ax_idx])

        ax.tick_params(labelsize=20)
        ax.grid()
        if ax_idx == 2:
            ax.set_ylim([-3, 3])

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels(x_ticklabels)


    print('here')


def axvline_axhline():
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    # ax.axvline(x=1, color='black', linewidth=1)
    # ax.axvline(x=1, ymax=0.8, ymin=0.2, color='black', linewidth=1)

    ax.axhline(y=1, color='black', linewidth=1)
    ax.axhline(y=1, xmax=0.8, xmin=0.2, color='black', linewidth=1)

    x = np.linspace(-4*np.pi, 4*np.pi, 200)
    sin = np.sin(x)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, sin)
    ax.axhline(y=1, ls=':', lw=1, color='gray')
    ax.axhline(y=-1, ls=':', lw=1, color='gray')


def legend_test():
    np.random.seed(0)

    n_data = 100
    random_noise1 = np.random.normal(0, 1, (n_data,))
    random_noise2 = np.random.normal(0, 1, (n_data,))
    random_noise3 = np.random.normal(0, 1, (n_data,))

    fix, ax = plt.subplots(figsize=(10, 7))
    ax.tick_params(labelsize=20)

    ax.plot(random_noise1, label='random noise1')
    ax.plot(random_noise2, label='random noise2')
    ax.plot(random_noise3, label='random noise3')

    ax.legend()
    # ax.legend(fontsize=20, loc='upper right')
    # ax.legend(fontsize=20, bbox_to_anchor=(1, 0.5), loc='center left')
    ax.legend(fontsize=20, bbox_to_anchor=(0, 0), loc='upper right')


def bbox_test():
    PI = np.pi
    t = np.linspace(-4*PI, 4*PI, 300)
    sin = np.sin(t)

    fig, ax = plt.subplots(figsize=(10, 10))

    for ax_idx in range(12):
        label_template = 'added by {}'
        ax.plot(t, sin+ax_idx, label=label_template.format(ax_idx))

    ax.legend(fontsize=15, ncol=4, bbox_to_anchor=(0.5, -0.05), loc='upper center')
    fig.tight_layout()


def violin_plot_test():
    np.random.seed(0)

    fig, ax = plt.subplots(figsize=(7, 7))

    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(5, 2, 200)
    data3 = np.random.normal(13, 3, 300)

    xticks = np.arange(3)

    # ax.violinplot([data1, data2, data3], showmeans=True, positions=xticks)
    violin = ax.violinplot([data1, data2, data3], positions=xticks,
                  quantiles=[[0.25, 0.75], [0.1, 0.9], [0.3, 0.7]])

    ax.set_xticks(xticks)
    ax.set_xticklabels(['setosa', 'versicolor', 'virginica'])
    ax.set_xlabel('Species', fontsize=15)
    ax.set_ylabel('Values', fontsize=15)

    violin['bodies'][0].set_facecolor('blue')
    violin['bodies'][1].set_facecolor('red')
    violin['bodies'][2].set_facecolor('green')

    violin['cbars'].set_edgecolor('gray')
    violin['cmaxes'].set_edgecolor('gray')
    violin['cmins'].set_edgecolor('gray')
    # violin['cmeans'].set_edgecolor('gray')



if __name__ == '__main__':
    # fig_test1()
    # fig_test2()
    # fig_test3()
    # fig_test4()
    # fig_test5()
    # fig_test6()
    # color_test()
    # scatter_plot()
    # scatter_plot2()
    # time_series()
    # time_series2()
    # sin_graph()
    # trigonometric_functions()
    # trigonometric_functions2()
    # axvline_axhline()
    # legend_test()
    # bbox_test()
    violin_plot_test()
    plt.show()


# object : 데이터 + 기능

