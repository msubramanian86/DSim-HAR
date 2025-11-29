
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Import load_data function
from load_data import load_data

def preprocess_data(filename, dataset_path='data/', test_size=0.2, random_state=42):
    """
    Preprocess the data by normalizing and splitting it into train and test sets.

    Args:
    filename (str): The name of the dataset file to preprocess.
    dataset_path (str): Path to the dataset (default 'data/').
    test_size (float): Proportion of the data to be used for testing (default 0.2).
    random_state (int): Random state for reproducibility (default 42).

    Returns:
    tuple: Tuple containing training features, test features, training labels, and test labels.
    """
    # Load data
    data = load_data(filename, dataset_path)

    # Assuming 'Activity' column is the label, separate features and labels
    X = data.drop(columns=['Activity'])
    y = data['Activity']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
