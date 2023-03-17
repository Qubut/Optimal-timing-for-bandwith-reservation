"""
This module contains a class for creating plots for different evaluation metrics used in a machine learning model. It uses the numpy and matplotlib.pyplot libraries.

Class:
- Plotter: A class that can create plots for input interval length, optimal level of risk, and evaluation metrics like validation mean absolute error, mean absolute percentage error, and root mean square error.

Dependencies:
- numpy
- matplotlib.pyplot
"""


import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """
    A class for creating plots for different evaluation metrics used in a machine learning model.
    """

    def __init__(self, font_size=12, resolution=300):
        """
        Initializes the Plotter object.

        Args:
        font_size (int): The font size to be used for the plot labels.
        resolution (int): The resolution of the plot image.
        """
        self.font_size = font_size
        self.resolution = resolution
        plt.rcParams.update({'font.size': font_size})

    def plot_iil(self, lstm_iil, tr_iil):
        """
        Creates a plot for the input interval length.

        Args:
        lstm_iil (list): The LSTM model's input interval length.
        tr_iil (list): The Transformer model's input interval length.

        Returns:
        None
        """
        plt.figure()
        plt.title('Test Mean Absolute Error')
        plt.ylabel('Test MAE')
        plt.xlabel('Input Interval Length')
        plt.plot(np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24]), lstm_iil, 'v-', label='LSTM')
        plt.plot(np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24]), tr_iil, 'v-', label='Transformers')
        plt.legend()
        plt.savefig('input interval length.png', dpi=self.resolution)

    def plot_greedy(self, greedy):
        """
        Creates a plot for the optimal level of risk.

        Args:
        greedy (list): The optimal level of risk values.

        Returns:
        None
        """
        plt.figure()
        plt.title('Optimal Level of Risk')
        plt.ylabel('Test MAE')
        plt.xlabel('c Value')
        plt.autoscale(axis='x', tight=True)
        plt.plot(np.arange(0, 50, 5), greedy, 'v-')
        plt.savefig('greedy.png', dpi=self.resolution)

    def plot_metrics(self, lstm, transformer):
        """
        Creates plots for different evaluation metrics.

        Args:
        lstm (list): The LSTM model's evaluation metrics.
        transformer (list): The Transformer model's evaluation metrics.

        Returns:
        None
        """
        plt.figure()
        plt.title('Validation Mean Absolute Error')
        plt.ylabel('Validation MAE')
        plt.xlabel('Epoch')
        plt.autoscale(axis='x',tight=True)
        plt.plot(lstm[:,0], label='LSTM')
        plt.plot(transformer[:,0], label='Transformers')
        plt.legend()
        plt.savefig('MAE.png', dpi=self.resolution)

        plt.figure()
        plt.title('Validation Mean Absolute Percentage Error')
        plt.ylabel('Validation MAPE')
        plt.xlabel('Epoch')
        plt.autoscale(axis='x',tight=True)
        plt.plot(lstm[:,1], label='LSTM')
        plt.plot(transformer[:,1], label='Transformers')
        plt.legend()
        plt.savefig('MAPE.png', dpi=self.resolution)

        plt.figure()
        plt.title('Validation Root Mean Square Error')
        plt.ylabel('Validation RMSE')
        plt.xlabel('Epoch')
        plt.autoscale(axis='x',tight=True)
        plt.plot(lstm[:,2], label='LSTM')
        plt.plot(transformer[:,2], label='Transformers')
        plt.legend()
        plt.savefig('RMSE.png', dpi=self.resolution)
    def plot_times(self, lstm_times, tr_times, lstm, transformer):
        """
        Plots the validation mean absolute error of LSTM and Transformers models over time.

        Args:
            lstm_times (numpy.ndarray): Array of time points for LSTM model.
            tr_times (numpy.ndarray): Array of time points for Transformers model.
            lstm (numpy.ndarray): Array of validation MAE for LSTM model.
            transformer (numpy.ndarray): Array of validation MAE for Transformers model.
        
        Returns:
        None
        """
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 15
        fig_size[1] = 5
        plt.rcParams["figure.figsize"] = fig_size

        plt.figure()
        plt.title('Validation Mean Absolute Error')
        plt.ylabel('Validation MAE')
        plt.xlabel('Seconds')
        plt.autoscale(axis='x',tight=True)
        plt.plot(lstm_times, lstm[:,0], label='LSTM')
        plt.plot(tr_times, transformer[:,0], label='Transformers')
        plt.legend()
        plt.savefig('MAE2.png', dpi=self.resolution)