import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go

raw_data = fdr.DataReader('005930', '2018')
print(raw_data.head())

# split x, y
X = raw_data.drop('Close', axis=1)
y = raw_data['Close'].values.reshape(-1, 1)

# Minmaxscaling
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)
print(f"{X_scaled.shape=}, {y_scaled.shape=}")

WINDOW_SIZE = 5 # 5일을 보고 1일을 예측
X_train = []
for index in range(len(X_scaled)-WINDOW_SIZE):
    X_train.append(X_scaled[index:index+WINDOW_SIZE])
X_train = np.array(X_train)
y_train = y_scaled[WINDOW_SIZE:]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

train_dataset = TensorDataset(X_train, y_train)

# 데이터셋 만들기
dataset = TensorDataset(X_train, y_train)
print(dataset.tensors[0].shape) # X_train
print(dataset.tensors[1].shape) # y_train

# Set Validation data
VALIDATION_RATE = 0.2
train_index, validation_index = train_test_split(range(len(train_dataset)), test_size=VALIDATION_RATE)

# Set Dataset
train_dataset = Subset(dataset, train_index)
validation_dataset = Subset(dataset, validation_index)

# Set Batches
BATCH_SIZE = 128
train_batches = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_batches = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Modeling
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # 출력 텐서 제공
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :] # many to one
        y = self.fc(output)
        return y


INPUT_SIZE = 5
HIDDEN_SIZE = 32
N_LAYERS = 2
DROPOUT = 0.15

model = LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

EPOCHS = 100
EARLY_STOP_THRESHOLD = 30
train_losses, validation_losses = list(), list()

for epoch in range(EPOCHS):
    epoch_train_loss, epoch_val_loss = 0., 0.
    lowest_loss = np.inf
    # train
    model.train()
    for X, y in train_batches:
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss = epoch_train_loss / len(train_batches)
    train_losses.append(epoch_train_loss)

    # evaluate
    model.eval()
    with torch.no_grad():
        for X, y in validation_batches:
            pred = model(X)
            loss = criterion(pred, y)
            epoch_val_loss += loss.item()

        epoch_val_loss = epoch_val_loss / len(validation_batches)
        validation_losses.append(epoch_val_loss)

        if epoch_val_loss < lowest_loss:
            lowest_loss = epoch_val_loss
            lowest_epoch = epoch
            best_model = deepcopy(model.state_dict())

        else:
            if (EARLY_STOP_THRESHOLD > 0) & (lowest_epoch + EARLY_STOP_THRESHOLD < epoch):
                print("Early Stopping..")
                print(f"epoch: {epoch}")
                break

    if epoch % 5 == 0:
        print(f"epoch: {epoch}/{EPOCHS} tr_loss: {epoch_train_loss:.6f} val_loss: {epoch_val_loss:.6f}")


# model validation
final_model = LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT)
final_model.load_state_dict(best_model)
test_batches = DataLoader(train_dataset, batch_size=BATCH_SIZE)

pred_list, actual_list = list(), list()
final_model.eval()
with torch.no_grad():
    for X, y in test_batches:
        pred = final_model(X)
        pred_list.append(pred)
        actual_list.append(y)


preds = torch.cat(pred_list, 0)
actuals = torch.cat(actual_list, 0)

predict_data = y_scaler.inverse_transform(preds)
real_data = y_scaler.inverse_transform(actuals)

# RMSE
rmse = mean_squared_error(real_data, predict_data) ** 0.5
print(f"{rmse=}")

concat_data = np.concatenate((predict_data, real_data), axis=1)
result_df = pd.DataFrame(concat_data, columns=['predict', 'real'])
print("=============result=============")
print(result_df)

# visualize
fig = go.Figure()
fig.add_trace(go.Scatter(x=result_df.index, y=result_df.predict, mode='lines', name='predict'))
fig.add_trace(go.Scatter(x=result_df.index, y=result_df.real, mode='lines', name='real'))
fig.show()

