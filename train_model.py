
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np

# Define a simple CNN model for HAR
class HAR_CNN(nn.Module):
    def __init__(self):
        super(HAR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(9, 64, kernel_size=3, stride=1, padding=1)  # Assuming 9 sensor channels
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 10, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 12)  # 12 activity classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 10)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(X_train, X_test, y_train, y_test, epochs=10, batch_size=64, learning_rate=0.001):
    """
    Train the HAR model on the given data.

    Args:
    X_train (numpy.ndarray): Training features.
    X_test (numpy.ndarray): Test features.
    y_train (numpy.ndarray): Training labels.
    y_test (numpy.ndarray): Test labels.
    epochs (int): Number of epochs (default is 10).
    batch_size (int): Batch size for training (default is 64).
    learning_rate (float): Learning rate (default is 0.001).

    Returns:
    model: Trained model.
    """
    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = HAR_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return model
