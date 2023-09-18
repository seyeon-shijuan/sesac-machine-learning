import matplotlib.pyplot as plt
import numpy as np


def base_mat():
    # matplotlib 기본 사용법
    fig = plt.figure()
    fig = plt.figure(figsize=(7, 7))
    fig = plt.figure(figsize=(7, 7), facecolor='linen')
    ax = fig.add_subplot()
    ax.plot([2, 3, 1])
    ax.scatter([2, 3, 1], [2, 3, 4])

    # 폰트
    figsize = (7, 7)
    fig, ax = plt.subplots(figsize=figsize)
    # fig 제목
    fig.suptitle("Title of a Figure", fontsize=30, fontfamily='monospace')

    # ax 제목
    ax.set_title("Title of a ax", fontsize=30, fontfamily='monospace')

    # 라벨 이름
    ax.set_xlabel("X label", fontsize=20)
    ax.set_ylabel("Y label", fontsize=20)

    fig.suptitle("Title of a Figure", fontsize=30, color='darkblue', alpha=0.9)
    ax.set_xlabel("X label", fontsize=20, color='darkblue', alpha=0.7)
    ax.set_ylabel("Y label", fontsize=20, color='darkblue', alpha=0.7)

    fig.tight_layout()


def my_twinx():
    # twinx: 한 그래프 공간에 2개 만들기
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()

    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    ax2.set_ylim([0, 0.1])

    ax1.set_title("Twinx Graph", fontsize=30)
    ax1.set_ylabel("Data1", fontsize=20)
    ax2.set_ylabel("Data2", fontsize=20)

    # ax1.tick_params(labelsize=20, length=10, width=3, bottom=False,
    #                 labelbottom=False, top=True, labeltop=True,
    #                 right=True, labelright=True)

    # ax1.tick_params(axis='x', labelsize=20, length=10, width=3, rotation=30)
    ax1.tick_params(axis='y', labelsize=20, length=10, width=3, rotation=50)

    fig.tight_layout()


def txt_alignment():
    figsize = (7, 7)
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    ax.grid()
    # labelsize 라벨 폰트
    ax.tick_params(axis='both', labelsize=15)
    ax.text(x=0, y=0, s='hello', fontsize=30)

    ax.text(x=0.5, y=0, s='hello2', fontsize=30)
    ax.text(x=0.5, y=-0.5, s='hello3', fontsize=30)


if __name__ == '__main__':
    # base_mat()
    # my_twinx()
    txt_alignment()
    plt.show()


