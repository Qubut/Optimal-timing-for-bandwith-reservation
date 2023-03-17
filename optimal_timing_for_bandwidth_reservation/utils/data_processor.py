"""

A module for processing data for time series prediction using neural networks.

Dependencies:

    torch
    pandas
    sklearn

Classes:

    DataProcessor: A class for loading and processing data for time series prediction.

"""

import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:

    """
    A class for loading and processing data for time series prediction.

    Args:
        train_file (str): The file path for the training data.
        test_file (str): The file path for the testing data.
        delta (int): The time step for the input sequence.
        device (str): The device to use for computation (e.g., 'cpu', 'cuda').

    Attributes:
        train_file (str): The file path for the training data.
        test_file (str): The file path for the testing data.
        delta (int): The time step for the input sequence.
        scaler (sklearn.preprocessing.MinMaxScaler): The scaler for normalizing the data.
        train_data (numpy.ndarray): The normalized training data.
        test_data (numpy.ndarray): The normalized testing data.

    """
    def __init__(self, train_file: str, test_file: str, delta: int, device: str):
        self.train_file = train_file
        self.test_file = test_file
        self.delta = delta
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_data = None
        self.test_data = None
        self.device = device

    def load_data(self):
        """
        Load and normalize the training and testing data.
        """
        df = pd.read_csv(self.train_file, sep=',')
        load = df['Lane 1 Flow (Veh/5 Minutes)'].values.astype(float)
        hourly_load = load.reshape((-1, 12)).mean(1)
        self.train_data = self.scaler.fit_transform(hourly_load.reshape(-1, 1))

        df2 = pd.read_csv(self.test_file, sep=',')
        load2 = df2['Lane 1 Flow (Veh/5 Minutes)'].values.astype(float)
        hourly_load2 = load2.reshape((-1, 12)).mean(1)
        self.test_data = self.scaler.transform(hourly_load2.reshape(-1, 1))

    def create_inout_sequences(self):
        """
        Create input-output sequences for training.
        
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences.
        """
        train_data = self.get_train_data()
        test_data = self.get_test_data()
        inout_seq = []
        L = len(self.train_data)
        for i in range(L - self.delta):
            train_seq = train_data[i:i + self.delta]
            train_label = train_data[i + self.delta:i + self.delta + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def get_test_data(self):
        """
        Get the normalized testing data as a PyTorch tensor.

        Returns:
            torch.Tensor: The normalized testing data.
        """
        return torch.FloatTensor(self.test_data).view(-1, self.delta).to(self.device)

    def get_train_data(self):
        """
        Get the normalized training data as a flattened PyTorch tensor.

        Returns:
            torch.Tensor: The normalized training data.
        """
        return torch.FloatTensor(self.train_data).flatten()



