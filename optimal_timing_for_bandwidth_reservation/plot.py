"""
This module loads data and calls the Plotter class to create visualizations for the experiment results.

Dependencies:
    numpy
    matplotlib
    visualisations.plotting

Output:
Creates multiple visualizations using the Plotter class to visualize the experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from visualisations.plotting import Plotter

if __name__ == "__main__":
    plotter = Plotter()

    lstm_iil = np.load("out/lstm/lstm_iil.npy")
    tr_iil = np.load("out/transformer/tr_iil.npy")

    lstm_acc = np.load("out/lstm/lstm_acc.npy")
    transformer_acc = np.load("out/transformer/tr_acc.npy")

    lstm_val_iil = np.load("out/lstm/lstm_validation_iil.npy")
    tr_val_iil = np.load("out/transformer/tr_validation_iil.npy")

    lstm_times = np.load("out/lstm/lstm_times.npy")
    tr_times = np.load("out/transformer/tr_times.npy")

    plotter.plot_iil(lstm_iil, tr_iil, title="Training Losses")
    plotter.plot_iil(lstm_val_iil, tr_val_iil, title="Validation Losses")
    plotter.plot_times(lstm_times, tr_times, lstm_acc, transformer_acc)
    plotter.plot_metrics(lstm_acc, transformer_acc)

    plt.show()
