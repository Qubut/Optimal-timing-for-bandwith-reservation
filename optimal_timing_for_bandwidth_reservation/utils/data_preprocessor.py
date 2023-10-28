"""
A module for processing data for time series prediction using neural networks.

Dependencies:
    torch
    dask
    pandas
    sklearn

Classes:
    DataPreProcessor: A class for loading and processing data for time series prediction.
"""

from typing import List, Tuple
import torch
import dask.dataframe as dd
from .scalers import RollingWindowScaler
from config.params import Params
import numpy as np

params = Params()


class DataPreProcessor:
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
        self.data_files = data_files
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.delta = delta
        self.scaler = RollingWindowScaler(window_size)
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.device = params.DEVICE
        self.num_providers = len(data_files)

    def load_data(self):
        def process_file(idx_file_tuple: Tuple[int, str]) -> dd.DataFrame:
            idx, file = idx_file_tuple

            # Read the CSV, assuming every file might have a header.
            df = dd.read_csv(file, sep=",", parse_dates=["Date"])

            # Apply timezone localization based on Region
            df["Date"] = df["Date"].where(
                df["Region"] == "us-west-1c", df["Date"].dt.tz_localize("US/Pacific")
            )
            df["Date"] = df["Date"].where(
                df["Region"] == "us-west-1b", df["Date"].dt.tz_localize("US/Pacific")
            )

            df = df.drop(columns=["Instance Type", "Region"])
            return df.rename(columns={"Price": f"Price_{idx}"}).set_index("Date")

        dfs = dd.from_delayed([dd.delayed(process_file)(item) for item in enumerate(self.data_files)])
        
        self.main_df = dfs.compute()
        self.main_df = self.main_df.fillna(method="ffill").fillna(method="bfill")

        # Convert timezone-aware datetime to timezone-naive datetime
        self.main_df["Date"] = self.main_df["Date"].dt.tz_localize(None)
        self.main_df["timestamp_unix"] = self.main_df["Date"].astype("int64") // 10**9

        # Normalize the prices in-place
        price_columns = [col for col in self.main_df.columns if "Price_" in col]
        for col in price_columns:
            self.main_df[col] = (
                self.main_df[col]
                .to_dask_array(lengths=True)
                .map_blocks(self.scaler.transform, dtype=float)
            )

        train_size = int(len(self.main_df) * self.train_ratio)
        self.val_size = int(train_size * self.validation_ratio)
        self.train_size = train_size - self.val_size

    def _create_inout_sequences(
        self, data: np.ndarray
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        inout_seq = []
        length = len(data)
        for i in range(length - self.delta):
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
        self.test_data = self.get_test_sequences(
            self.main_df.loc[self.train_size + self.val_size :].compute().to_numpy()
        )
        return self._create_inout_sequences(self.test_data)

    def get_train_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        self.train_data = self.get_train_sequences(
            self.main_df.loc[: self.train_size - 1].compute().to_numpy()
        )

        return self._create_inout_sequences(self.train_data)

    def get_val_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        self.val_data = self.get_val_sequences(
            self.main_df.loc[self.train_size : self.train_size + self.val_size - 1]
            .compute()
            .to_numpy()
        )
        return self._create_inout_sequences(self.validation_data)
