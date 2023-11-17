from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD


# GPU 사용 가능 -> True, GPU 사용 불가 -> False
# print(torch.cuda.is_available())

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"curr device = {DEVICE}")
def get_binary_linear_dataset():
    n_samples = 100
    X, y = make_blobs(n_samples=n_samples, centers=2,
                      n_features=2, cluster_std=0.5)

    fig, ax = plt.subplots(figsize=(5, 5))

    X_pos, X_neg = X[y == 1], X[y == 0]
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red')
    ax.tick_params(labelsize=15)
    # ax.scatter(X[:, 0], X[:, 1], c=y)
    fig.tight_layout()
    plt.show()

    return X, y


def get_binary_moon_dataset():
    n_samples = 300
    X, y = make_moons(n_samples=n_samples, noise=0.2)

    X_pos, X_neg = X[y == 1], X[y == 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color='red')
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.show()

    return X, y


def get_linear_dataloader():
    n_samples = 100
    X, y = make_blobs(n_samples=n_samples, centers=2,
                          n_features=2, cluster_std=0.7)

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))

    BATCH_SIZE = 32
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for X_, y_ in dataloader:
        print(type(X_), X_.shape, X_.dtype)
        print(type(y_), y_.shape, y_.dtype)


get_linear_dataloader()

def get_moon_dataloader():
    n_samples = 300
    BATCH_SIZE = 16
    X, y = make_moons(n_samples=n_samples, noise=0.2)

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for X_, y_ in dataloader:
        print(type(X_), X_.shape, X_.dtype)
        print(type(y_), y_.shape, y_.dtype)
        break


def linear_shape():
    fc = nn.Linear(in_features=8, out_features=4)
    print(fc.weight.shape) # torch.Size([4, 8])
    print(fc.bias.shape) # torch.Size([4])


def linear_shape2():
    test_input = torch.randn(size=(16, 8))
    fc = nn.Linear(in_features=8, out_features=4)
    test_output = fc(test_input)

    print(f"test input: {test_input.shape}") # test input: torch.Size([16, 8])
    print(f"test output: {test_output.shape}") # test input: torch.Size([16, 4])


def after_sigmoid():
    test_input = torch.randn(size=(2, 3))
    sigmoid = nn.Sigmoid()
    test_output = sigmoid(test_input)

    print("======= Test Input ========")
    print(test_input)

    print("======= nn.Sigmoid Output ========")
    print(test_output)

    print("======= manual computation ========")
    print(1 / (1 + torch.exp(-test_input)))

    '''
    ======= Test Input ========
    tensor([[-0.1151, -0.4276,  1.0766],
            [ 0.3289,  0.3325,  0.4018]])
    ======= nn.Sigmoid Output ========
    tensor([[0.4713, 0.3947, 0.7459],
            [0.5815, 0.5824, 0.5991]])
    ======= manual computation ========
    tensor([[0.4713, 0.3947, 0.7459],
            [0.5815, 0.5824, 0.5991]])
    '''


def after_bceloss():
    test_pred = torch.tensor([0.8])
    test_y = torch.tensor([1.])

    loss_function = nn.BCELoss()
    test_output = loss_function(test_pred, test_y)

    print("======= Test Input ========")
    print(f"{test_pred=}")
    print(f"{test_y=}")
    print("======= nn.Sigmoid Output ========")
    print(f"{test_output=}")
    print("======= manual computation ========")
    print(-(test_y * torch.log(test_pred) + (1 - test_y) * torch.log(1 - test_pred)))
    '''
    ======= Test Input ========
    test_pred=tensor([0.8000])
    test_y=tensor([1.])
    ======= nn.Sigmoid Output ========
    test_output=tensor(0.2231)
    ======= manual computation ========
    tensor([0.2231])
    '''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=5)
        self.fc1_activation = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=2)
        self.fc2_activation = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=2, out_features=1)
        self.fc3_activation = nn.Sigmoid()
    
    def forward(self, x):
        # 학습을 위해 z1, y1.. 으로 표기함
        # 이후에는 메모리 절약을 위해 x로 통일
        z1 = self.fc1(x); print(f"{z1.shape=}")
        y1 = self.fc1_activation(z1)

        z2 = self.fc2(y1); print(f"{z2.shape=}")
        y2 = self.fc2_activation(z2)

        z3 = self.fc3(y2); print(f"{z3.shape=}")
        logits = self.fc3_activation(z3); print(f"{logits.shape=}")
        return logits


def model_trial():
    n_samples = 100
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=10, cluster_std=0.7)
    X = torch.FloatTensor(X)
    print(f"{X.shape=}")

    model = Model()
    logits = model.forward(X)

    '''
    X.shape=torch.Size([100, 10])
    z1.shape=torch.Size([100, 5])
    z2.shape=torch.Size([100, 2])
    z3.shape=torch.Size([100, 1])
    logits.shape=torch.Size([100, 1])
    '''


# make blobs 만들어서 학습시키기
class SimpleModel(nn.Module):
    def __init__(self, n_features):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = x.view(-1) # (B,1)을 (B,)의 벡터로 squeeze
        return x

def simple_model_training():
    # Data
    N_SAMPLES = 100
    BATCH_SIZE = 32
    n_features = 2
    X, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=n_features, cluster_std=0.5)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Training
    LR = 0.1
    model = SimpleModel(n_features=n_features)
    model.to(DEVICE)
    optimizer = SGD(model.parameters(), lr=LR)
    loss_function = nn.BCELoss()
    EPOCHS = 10
    losses, accs = list(), list()

    for epoch in range(EPOCHS):
        epoch_loss, n_corrects = 0., 0
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            loss = loss_function(pred, y)

            optimizer.zero_grad()
            loss.backward() # 모델의 파라미터가 학습되는 부분
            optimizer.step()

            # Batch Size의 loss로 변환하여 누적
            epoch_loss += loss.item() * len(X)
            pred = (pred > 0.5).type(torch.float)
            n_corrects += (pred == y).sum().item()

        epoch_loss /= N_SAMPLES
        epoch_accr = n_corrects / N_SAMPLES
        print(f"Epoch: {epoch + 1}", end="\t")
        print(f"Loss: {epoch_loss:.4f}", end="\t")
        print(f"Accuracy: {epoch_accr:.4f}")
        losses.append(epoch_loss)
        accs.append(epoch_accr)

    fig, axes = plt.subplots(2, 1, figsize=(7, 3))
    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("BCELoss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)
    fig.suptitle("1-Layer Model Eval Metrics by Epoch", fontsize=16)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    simple_model_training()