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
    lstm = np.load("out/lstm/lstm_acc.npy")
    transformer = np.load("out/transformer/tr_acc.npy")
    lstm_times = np.load("out/lstm/lstm_times.npy")
    tr_times = np.load("out/transformer/tr_times.npy")
    greedy = np.load("out/greedy.npy")
    plotter.plot_iil(lstm_iil, tr_iil)
    plotter.plot_times(lstm_times, tr_times, lstm, transformer)
    plotter.plot_greedy(greedy)
    plotter.plot_metrics(lstm, transformer)
