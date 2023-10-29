import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    roc_auc_score, silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import auc

from project1.pjt_02_mlp_train import MLP, test_dataloader
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

input_size = (10127, 37)  # 입력 특성의 크기
hidden_size1 = 64  # 첫 번째 은닉 레이어의 크기
hidden_size2 = 32  # 두 번째 은닉 레이어의 크기
hidden_size3 = 16   # 세 번째 은닉 레이어의 크기
output_size = 1     # 출력 레이어의 크기
learning_rate = 0.001

# Load a pre-trained model
pretrained_model_filename = 'model_epoch_10000.pt'  # Change the filename to the one you want to load

# Create an instance of your model
model = MLP(input_size[1], hidden_size1, hidden_size2, hidden_size3, output_size)

# Load the model's state dictionary
model.load_state_dict(torch.load(pretrained_model_filename))

# Set the model to evaluation mode if you intend to use it for inference
model.eval()

# Now you can use the loaded model for inference or further training if needed


# Assuming that your model and test data are already defined as in your previous code
model.eval()  # Set the model to evaluation mode

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
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print Silhouette Score
print(f"Silhouette Score: {silhouette_avg:.2f}")
