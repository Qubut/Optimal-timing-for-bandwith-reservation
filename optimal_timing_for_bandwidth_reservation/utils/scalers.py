import numpy as np
import pandas as pd


class RollingWindowScaler:
    def __init__(self, window_size):
        self.window_size = window_size
        self.rolling_mean = None
        self.rolling_std = None

    def transform(self, data):
        self.rolling_mean = (
            pd.Series(data.flatten()).rolling(window=self.window_size).mean().values
        )

        self.rolling_std = (
            pd.Series(data.flatten()).rolling(window=self.window_size).std().values
        )

        self.rolling_mean[np.isnan(self.rolling_mean)] = self.rolling_mean[
            self.window_size - 1
        ]
        self.rolling_std[np.isnan(self.rolling_std)] = self.rolling_std[
            self.window_size - 1
        ]
        scaled_data = (data - self.rolling_mean.reshape(-1, 1)) / (
            self.rolling_std.reshape(-1, 1) + 1e-8
        )

        return scaled_data.reshape(-1, 1)

    def inverse_transform(self, data):
        return (data * (self.rolling_std + 1e-8) + self.rolling_mean).reshape(-1, 1)
