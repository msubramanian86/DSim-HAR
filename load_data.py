
import pandas as pd
import os

def load_data(filename, dataset_path='data/'):
    """
    Function to load a dataset from the given CSV file.

    Args:
    filename (str): The name of the file to load (e.g., 'pamap2.csv')
    dataset_path (str): The path to the dataset folder (default is 'data/')

    Returns:
    pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    file_path = os.path.join(dataset_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {filename} not found in {dataset_path}.")

    data = pd.read_csv(file_path)
    return data
