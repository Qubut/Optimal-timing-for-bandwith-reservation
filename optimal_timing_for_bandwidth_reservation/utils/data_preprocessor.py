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
        dfs = []

        for idx, file in enumerate(self.data_files):
            df = dd.read_csv(file, sep=",", parse_dates=["Date"])
            df = df.drop(columns=["Instance Type", "Region"])
            df = df.rename(columns={"Price": f"Price_{idx}"}).set_index("Date")
            dfs.append(df)

        main_df = dd.concat(dfs, axis=1, interleave_partitions=True).reset_index()

        main_df = main_df.fillna(method="ffill").fillna(method="bfill")

        # Convert timestamp to unix timestamp (seconds since epoch)
        main_df["timestamp_unix"] = (
            main_df["Date"].astype("M8[ns]").astype("int64") // 10**9
        )
        features = main_df[["timestamp_unix"]].to_dask_array(lengths=True)
        prices = main_df.drop(columns=["Date", "timestamp_unix"]).to_dask_array(
            lengths=True
        )

        prices_normalized = (
            self.scaler.transform(prices.compute().ravel())
            .reshape(prices.shape)
            .compute()
        )
        data = dd.from_array(dd.concatenate([features, prices_normalized], axis=1))

        train_size = dd.from_delayed(len(data) * self.train_ratio).compute()
        val_size = dd.from_delayed(train_size * self.validation_ratio).compute()
        _train_size = train_size - val_size

        self.train_data = data.loc[: _train_size - 1].compute().to_numpy()
        self.validation_data = (
            data.loc[_train_size : _train_size + val_size - 1].compute().to_numpy()
        )
        self.test_data = data.loc[_train_size + val_size :].compute().to_numpy()

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
        return self._create_inout_sequences(self.test_data)

    def get_train_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self._create_inout_sequences(self.train_data)

    def get_validation_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self._create_inout_sequences(self.validation_data)
