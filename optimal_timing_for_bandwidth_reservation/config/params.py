from dataclasses import dataclass, field
from torch import device, cuda


@dataclass
class Params:
    DELTA = 8
    EPOCHS = 150
    BATCH_SIZE = 32
    FREQ_PRINTING = 10
    DEVICE = field(default_factory=lambda:device("cuda" if cuda.is_available() else "cpu"))
    N_HEAD=5
    N_LAYERS=5
    DROPOUT=0.25
    N_HIDDEN_L=100
    DATAFILES=field(default_factory=lambda:["./datasets/Dataset_NO1.csv", "./datasets/Dataset_NO2.csv", "./datasets/Dataset_NO3.csv"])
    LSTM_RESULTS_PATH="../out/lstm"
    TRANSFORMER_RESULTS_PATH="../out/transfomer"
    SAVED_MODELS_PATH="../out/models"
    RUNS_LOG_PATH="../runs"
    
    
