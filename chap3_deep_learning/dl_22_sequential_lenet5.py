from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm


## nn.Sequential로 변환
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.feature = nn.Sequential(OrderedDict([
                ('cnn1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)),
                ('cnn1_act', nn.Tanh()),
                ('avgpool1', nn.AvgPool2d(kernel_size=2, stride=2)),

                ('cnn2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
                ('cnn2_act', nn.Tanh()),
                ('avgpool2', nn.AvgPool2d(kernel_size=2, stride=2)),

                ('cnn3', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)),
                ('cnn3_act', nn.Tanh())

            ]))

        self.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=120, out_features=84)),
                ('fc1_act', nn.Tanh()),

                ('fc2', nn.Linear(in_features=84, out_features=10))
            ]))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def run_lenet():
    # config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(model)
    EPOCHS = 3
    LR = 0.001
    BATCH_SIZE = 64
    N_SAMPLES = 60000

    # MNIST config
    dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model = LeNet().to(DEVICE)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    losses, accs = list(), list()

    for e in range(EPOCHS):
        epoch_loss, n_corrects = 0., 0

        for X_, y_ in tqdm(dataloader):
            X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
            pred = model(X_)
            loss = loss_fn(pred, y_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            pred_cls = torch.argmax(pred, dim=1)
            n_corrects += (pred_cls == y_).sum().item()

        epoch_loss /= N_SAMPLES
        epoch_accr = n_corrects / N_SAMPLES

        print(f"epoch {e} : loss={epoch_loss.item():.4f}, accr={epoch_accr}")

    losses.append(epoch_loss)
    accs.append(epoch_accr)
    print(losses)
    print(accs)



if __name__ == '__main__':
    run_lenet()


