from dl_09_full_nn import Model
from dl_09_full_nn import BCELoss

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(preds, targets):
    rounded_preds = np.round(preds)
    correct = (rounded_preds == targets).sum().item()
    accuracy = correct / len(targets)
    return accuracy


def main_routine():
    N_SAMPLES = 100
    LR = 0.001
    EPOCHS = 30

    X, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=2,
                      cluster_std=0.5, random_state=0)

    '''Instantiation'''
    model = Model()
    loss_fn = BCELoss()

    epoch_accuracy = list()
    epoch_loss = list()

    for epoch in range(EPOCHS):
        pred_list = list()
        loss_list = list()

        for X_, y_ in zip(X, y):
            '''Training'''
            pred = model.forward(X_)
            loss, dJ_dpred = loss_fn(pred, y_)
            w, b = model.backward(dJ_dpred, LR)

            '''Metric(loss, accuracy) Calculations'''
            pred_list.append(pred)
            loss_list.append(loss)

        epoch_accuracy.append(calculate_accuracy(pred_list, y))
        epoch_loss.append(sum(loss_list)/len(loss_list))

    '''Result Visualization'''
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(epoch_loss, linestyle='-')
    axes[0].set_ylabel("BCELoss", fontsize=15)
    axes[1].plot(epoch_accuracy, linestyle='-')
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[1].set_xlabel("Epoch", fontsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_routine()