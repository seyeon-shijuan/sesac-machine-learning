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
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# all_models = models.list_models()

# 전체 layer를 frozen하기
for param in model.parameters():
    param.requires_grad = False

# 마지막 fc layer만 재학습하는 것으로 수정
for param in model.fc.parameters():
    param.requires_grad = True

# fc 수정
model.fc = (
    nn.Sequential(
        nn.Linear(model.fc.in_features, 10),
        nn.LogSoftmax(dim=1)
    )
)

model.to(DEVICE)

print(model)

# 2. Dataset
# 이미지 transform
transform = transforms.Compose([
    # 흑백이미지를 컬러로 바꾸기 (1-ch to 3-ch)
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

# train dataset에서 val 0.2 분리
train_idx, val_idx = train_test_split(range(len(train_dataset)),
                                      stratify=train_dataset.targets,
                                      test_size=0.2)

train_dataset = Subset(train_dataset, train_idx)
validation_dataset = Subset(train_dataset, val_idx)
# train 48000 validation 12000 test 10000

# DataLoader
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 5
N_SAMPLES = len(train_dataset)
N_TEST = len(test_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=LR)
# 다중분류에 NLL Loss 사용
criterion = nn.NLLLoss()


# Train
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

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss}, Accuracy: {epoch_accr}")
    # model save
    torch.save(model, f'resnet50_e0{epoch+1}.pt')


# model save
torch.save(model, 'resnet50_.pt')

# model load
model = torch.load('resnet50_.pt')
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

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    ax.axis('off')
    ax.imshow(wrong_input[idx][0, :, :].numpy(), cmap='gray')
    ax.set_title(f"pred: {str(wrong_preds_idx[idx].item())}, actual: {str(actual_preds_idx[idx].item())}")

plt.tight_layout()
plt.savefig('wrong_nums.jpg')
plt.show()
print('here')

