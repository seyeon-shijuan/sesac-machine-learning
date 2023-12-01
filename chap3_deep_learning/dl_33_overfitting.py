import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def get_sin_ds(n_samples=50, if_vis=False):
    # dataset
    x = np.random.uniform(-np.pi, np.pi, n_samples)
    y = np.sin(x) + 0.2 * np.random.randn(n_samples)

    # visualize
    if if_vis:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(x=x, y=y)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        for spine_loc, spine in ax.spines.items():
            if spine_loc in ['right', 'top']:
                spine.set_visible(False)

        plt.tight_layout()
        plt.savefig('sin_dataset.png')
        plt.show()

    return x, y


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SinRegresser(nn.Module):
    def __init__(self):
        super(SinRegresser, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    np.random.seed(8)
    x, y = get_sin_ds(n_samples=50, if_vis=False)
    DEVICE = get_device()
    print(f"curr device = {DEVICE}")
    print(f"{len(x)=} {len(y)=}")

