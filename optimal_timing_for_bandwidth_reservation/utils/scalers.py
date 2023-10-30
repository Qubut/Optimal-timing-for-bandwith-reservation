import dask.dataframe as dd
import pandas as pd


class RollingWindowScaler:
    def __init__(self, window_size):
        self.window_size = window_size
        self.rolling_mean = None
        self.rolling_std = None

    def transform(self, data_column):
        if isinstance(data_column, pd.Series):
            data_column = dd.from_pandas(data_column, npartitions=1)

        rolling_mean = data_column.rolling(window=self.window_size).mean()
        rolling_std = data_column.rolling(window=self.window_size).std()

        rolling_mean = rolling_mean.fillna(method="bfill").fillna(method="ffill")
        rolling_std = rolling_std.fillna(method="bfill").fillna(method="ffill")

        scaled_data = (data_column - rolling_mean) / (rolling_std + 1e-8)
        self.rolling_mean = rolling_mean
        self.rolling_std = rolling_std
        return scaled_data

    def inverse_transform(self, data_column):
        if isinstance(data_column, pd.Series):
            data_column = dd.from_pandas(data_column, npartitions=1)

        return data_column * (self.rolling_std + 1e-8) + self.rolling_mean
