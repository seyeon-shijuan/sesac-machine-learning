import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(torch.__version__)

df = pd.read_csv("../data/bankchurners_train.csv", index_col=0)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
encoded_data, labels = pd.factorize(y)
y = pd.Series(encoded_data)
col_names = X.columns.to_list()
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets for training and testing data
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# DataLoaders for training and testing data
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# MLP layer 3개
# 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, input_size, bias=False)
        # self.fc = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        attention_weights = F.softmax(self.W(input_data), dim=1)
        context_vector = torch.sum(attention_weights * input_data, dim=1)
        # x = self.fc(context_vector)
        # x = self.sigmoid(x)
        return context_vector


# 모델 및 손실 함수, 최적화기 설정
input_size = (10127, 37)  # 입력 특성의 크기
hidden_size1 = 64  # 첫 번째 은닉 레이어의 크기
hidden_size2 = 32  # 두 번째 은닉 레이어의 크기
hidden_size3 = 16   # 세 번째 은닉 레이어의 크기
output_size = 1     # 출력 레이어의 크기
learning_rate = 0.001

# model = MLP(input_size[1], hidden_size1, hidden_size2, hidden_size3, output_size)
model = AttentionModel(input_size[1], batch_size)  # 입력 데이터의 특성 수에 따라 적절한 입력 크기를 설정
model.to(device)

# criterion = nn.MSELoss()  # 이진 분류 손실 (Binary Cross-Entropy Loss)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.BCELoss()  # 이진 분류 손실 (Binary Cross-Entropy Loss)
criterion = nn.BCEWithLogitsLoss()  # 이진 분류를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
num_epochs = 10000


for epoch in range(num_epochs):
    correct = 0
    total = 0
    for inputs, labels in train_dataloader:
        # inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동
        optimizer.zero_grad()

        outputs = model(inputs)
        # outputs = outputs.squeeze(dim=1)
        loss = criterion(outputs, labels)  # labels를 float로 변환하여 손실 계산

        loss.backward()
        optimizer.step()

    # 매 에포크 끝에 손실 출력
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        predicted = (outputs > 0.5).int()  # 0.5 이상일 때 1로 분류
        total += labels.size(0)
        correct += (predicted.flatten() == labels).sum().item()

    accuracy = 100 * correct / total

    # 매 에포크 끝에 손실과 정확도 출력
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy:.2f}%')

    # Save the model every 1000 epochs
    if (epoch + 1) % 500 == 0:
        model_filename = f'./models/model_epoch_{epoch + 1}.pt'
        torch.save(model.state_dict(), model_filename)
        print(f'Model saved as {model_filename}')
        attention_weights = model.W.weight.data.cpu().numpy()

        plt.figure(figsize=(10, 2))
        plt.bar(range(input_size[1]), attention_weights[0])
        plt.title('Attention Weights')
        # plt.show()
        plt.savefig(f'./img/model_epoch_{epoch + 1}.png')




def validate_test_data(pretrained_model_filename):
    # Test
    # Load a pre-trained model
    # pretrained_model_filename = './models/model_epoch_4000.pt'

    # Create an instance of your model
    model = MLP(input_size[1], hidden_size1, hidden_size2, hidden_size3, output_size)
    model = model.to(device)

    # Load the model's state dictionary
    model.load_state_dict(torch.load(pretrained_model_filename))

    # Set the model to evaluation mode if you intend to use it for inference
    model.eval()

    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Lists to store raw probabilities for ROC curve
    raw_probabilities = []

    # Iterate through the test data
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze(dim=1)

        predicted = (outputs > 0.5).int()  # 0.5 이상일 때 1로 분류

        # Convert predictions and labels to numpy arrays
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        # Store the raw probabilities
        raw_probabilities.extend(outputs.cpu().detach().numpy())

    # Calculate Confusion Matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Calculate Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate Precision
    precision = precision_score(true_labels, predicted_labels)

    # Calculate Recall
    recall = recall_score(true_labels, predicted_labels)

    # Calculate F1 Score
    f1 = f1_score(true_labels, predicted_labels)

    # Calculate ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, raw_probabilities)
    roc_auc = auc(fpr, tpr)

    # Calculate Silhouette Score (Assuming you are doing clustering)
    # If you are not performing clustering, please remove this part
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)  # Scale the data if necessary

    # Use PCA for dimension reduction (change the number of components as needed)
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_scaled)

    # Assuming you are using K-Means clustering (change the number of clusters as needed)
    kmeans = KMeans(n_clusters=2, random_state=0)
    cluster_labels = kmeans.fit_predict(X_test_pca)
    silhouette_avg = silhouette_score(X_test_pca, cluster_labels)

    # Print the metrics
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Plot ROC Curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # # Print Silhouette Score
    # print(f"Silhouette Score: {silhouette_avg:.2f}")


for i in range(1000, 10000, 1000):
    model_frame = f'./models/model_epoch_{i}.pt'
    print(model_frame)
    validate_test_data(model_frame)
