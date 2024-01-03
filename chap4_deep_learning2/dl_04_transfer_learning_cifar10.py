import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE=}")

# 1. Model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# Softmax는 criterion에서 계산
model.fc = nn.Linear(model.fc.in_features, out_features=10)
model.to(DEVICE)


# 2. Dataset
# 이미지를 텐서(Tensor)로 변환
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 훈련용 데이터셋
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)

# 테스트용 데이터셋
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

# 클래스 레이블
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Dataloader
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 5
N_SAMPLES, N_TEST = len(train_dataset), len(test_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=LR)
# nn.CrossEntropyLoss는 내부에서 Log Softmax를 수행하고 nll_loss를 계산함
criterion = nn.CrossEntropyLoss()

# 3. Train
for epoch in range(EPOCHS):
    model.train()
    epoch_loss, n_corrects = 0., 0

    for X, y in tqdm(train_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X)
        pred_cls = torch.argmax(pred, dim=1)
        n_corrects += (pred_cls == y).sum().item()

    epoch_loss /= N_SAMPLES
    epoch_accr = n_corrects / N_SAMPLES

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss}, Accuracy: {epoch_accr}")
    # model save
    torch.save(model, f'resnet32_e0{epoch + 1}.pt')


# model save
torch.save(model, 'resnet32_.pt')

# model load
model = torch.load('resnet32_.pt')
model = model.to(DEVICE)

# evaluation
with torch.no_grad():
    total_loss, n_corrects, = 0., 0
    wrong_input, wrong_preds_idx, actual_preds_idx = list(), list(), list()

    model.eval()
    for X_, y_ in tqdm(test_dataloader):
        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)

        pred = model(X_)
        total_loss += criterion(pred, y_)
        pred_cls = torch.argmax(pred, dim=1)
        n_corrects += (pred_cls == y_).sum().item()

        wrong_idx = pred_cls.ne(y_).nonzero()[:, 0].cpu().numpy().tolist()
        for index in wrong_idx:
            wrong_input.append(X_[index].cpu())  # 잘못 예측한 X 값
            wrong_preds_idx.append(pred_cls[index].cpu())  # 잘못 예측한  Y값
            actual_preds_idx.append(y_[index].cpu())  # 실제 Y값

total_loss /= N_TEST
total_accr = n_corrects / N_TEST


# visualization

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    ax.axis('off')
    ax.imshow(wrong_input[idx][0, :, :].numpy(), cmap='gray')
    pred_ = classes[wrong_preds_idx[idx].item()]
    actual_ = classes[actual_preds_idx[idx].item()]
    ax.set_title(f"pred: {str(pred_)}, actual: {str(actual_)}", fontsize=9)


plt.savefig('wrong_images.jpg')
plt.show()
print('here')