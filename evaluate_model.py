
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
    model: The trained model.
    X_test (numpy.ndarray): Test features.
    y_test (numpy.ndarray): Test labels.

    Returns:
    None
    """
    # Convert to PyTorch tensor
    test_data = torch.Tensor(X_test)
    test_labels = torch.Tensor(y_test).long()

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, preds = torch.max(outputs, 1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, preds.numpy())
    print(f"Confusion Matrix:
{cm}")

    # Display classification report
    print(f"Classification Report:
{classification_report(y_test, preds.numpy())}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(12)  # Assuming 12 classes
    plt.xticks(tick_marks, range(12))
    plt.yticks(tick_marks, range(12))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
