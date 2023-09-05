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

from .scalers import RollingWindowScaler


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

    def __init__(
        self,
        data_file: str,
        device: str,
        delta: int = 100,
        train_ratio: float = 0.7,
        window_size=100,
    ):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.delta = delta
        self.scaler = RollingWindowScaler(window_size)
        self.train_data = None
        self.test_data = None
        self.device = device

    def load_data(self):
        """
        Load and normalize the training and testing data.
        """
        df = pd.read_csv(self.data_file, sep=",", parse_dates=[0])
        TIMESTAMP_COL = 0
        df = df.sort_values(by=df.columns[TIMESTAMP_COL])

        prices = df.iloc[:, -1].values.astype(float).reshape(-1, 1)
        prices_normalized = self.scaler.transform(prices)

        # Split data into training and test sets based on the train_ratio
        train_size = int(len(prices_normalized) * self.train_ratio)
        self.train_data = prices_normalized[:train_size]
        self.test_data = prices_normalized[train_size:]

    def _create_inout_sequences(self, data):
        """
        Create input-output sequences for training or testing.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences.
        """
        inout_seq = []
        length = int(len(data))
        delta = 100
        
        for i in range(length - delta):
            seq = (
                torch.FloatTensor(data[i : i + delta])
                .view(delta, -1, 1)
                .to(self.device)
            )
            label = (
                torch.FloatTensor(data[i + delta : i + delta + 1])
                .view(1, -1)
                .to(self.device)
            )
            inout_seq.append((seq, label))
        return inout_seq

    def get_test_sequences(self):
        """
        Get the sequences for testing.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences for testing.
        """
        return self._create_inout_sequences(self.test_data)

    def get_train_sequences(self):
        """
        Get the sequences for training.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences for training.
        """
        return self._create_inout_sequences(self.train_data)
