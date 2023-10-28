"""

A module for processing data for time series prediction using neural networks.

Dependencies:

    torch
    pandas
    sklearn

Classes:

    DataProcessor: A class for loading and processing data for time series prediction.

"""

from typing import List, Tuple
import numpy as np
import torch
import pandas as pd

from .scalers import RollingWindowScaler
from config import device


class DataProcessor:
    """
    A class for loading and processing data for time series prediction.

    Args:
        data_files (List[str]): The list of file paths for the datasets, one for each provider.
        delta (int): The time step for the input sequence.
        train_ratio (float): The train-test split ratio.

    Attributes:
        data_files (List[str]): The list of file paths for the datasets.
        delta (int): The time step for the input sequence.
        scaler (RollingWindowScaler): The scaler for normalizing the data.
        train_data (numpy.ndarray): The normalized training data.
        test_data (numpy.ndarray): The normalized testing data.
    """

    def __init__(
        self,
        data_files: List[str],
        delta: int = 32,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        window_size=100,
    ):
        self.data_filess = data_files
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.delta = delta
        self.scaler = RollingWindowScaler(window_size)
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.device = device
        self.num_providers = len(data_files)
    def load_data(self):
        main_df = pd.read_csv(self.data_files[0], sep=",", parse_dates=["Date"])
        main_df = main_df.rename(columns={"Price": "Price_0"})
        main_df.drop(columns=["Instance Type", "Region"], inplace=True, axis=1)

        for idx, file in enumerate(self.data_files[1:], 1):
            df = pd.read_csv(file, sep=",", parse_dates=["Date"])
            df.drop(columns=["Instance Type", "Region"], inplace=True, axis=1)
            df = df.rename(columns={"Price": f"Price_{idx}"})
            main_df = pd.merge(main_df, df, on="Date", how="outer")

        main_df.sort_values(by="Date", inplace=True)
        main_df.fillna(method="ffill", inplace=True)
        main_df.fillna(method="bfill", inplace=True)

        # Convert timestamp to unix timestamp (seconds since epoch)
        main_df["timestamp_unix"] = main_df["Date"].astype(np.int64) // 10**9
        features = main_df["timestamp_unix"].values.reshape(-1, 1)
        prices = main_df.iloc[
            :, 1:-1
        ].values  # Excluding the Date and the unix timestamp

        prices_normalized = self.scaler.transform(prices.ravel()).reshape(prices.shape)

        data = np.hstack([features, prices_normalized])

        train_size = int(len(data) * self.train_ratio)
        
        val_size = int(train_size * self.validation_ratio)
        _train_size = train_size - val_size
        
        self.train_data = data[:_train_size]
        self.validation_data = data[_train_size : _train_size + val_size]
        self.test_data = data[_train_size + val_size:]
        
    def _create_inout_sequences(
        self, data: np.ndarray
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create input-output sequences for training or testing.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences.
        """

        inout_seq = []
        length = len(data)

        for i in range(length - self.delta):
            # The input is just the timestamp, so only the first column is taken
            seq = (
                torch.FloatTensor(data[i : i + self.delta, 0])
                .view(self.delta, -1, 1)
                .to(self.device)
            )

            label = (
                torch.FloatTensor(data[i + self.delta : i + self.delta + 1, 1:])
                .view(1, -1, self.num_providers)
                .to(self.device)
            )

        inout_seq.append((seq, label))
        return inout_seq

    def get_test_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the sequences for testing.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences for testing.
        """
        return self._create_inout_sequences(self.test_data)

    def get_train_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the sequences for training.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences for training.
        """
        return self._create_inout_sequences(self.train_data)
    
    def get_validation_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the sequences for validation.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of input and output sequences for validation.
        """
        return self._create_inout_sequences(self.validation_data)