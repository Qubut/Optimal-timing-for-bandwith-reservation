from dataclasses import dataclass, field
from typing import List
from torch import device
from torch.cuda import is_available


@dataclass
class Params:
    DELTA: int = 8
    EPOCHS: int = 150
    BATCH_SIZE: int = 32
    FREQ_PRINTING: int = 10
    DEVICE: device = field(
        default_factory=lambda: device("cuda" if is_available() else "cpu")
    )
    N_HEAD: int = 5
    N_LAYERS: int = 5
    DROPOUT: float = 0.25
    N_HIDDEN_L: int = 100
    DATAFILES: List[str] = field(
        default_factory=lambda: [
            "../datasets/Dataset_NO1.csv",
            "../datasets/Dataset_NO2.csv",
            "../datasets/Dataset_NO3.csv",
        ]
    )
    LSTM_RESULTS_PATH: str = "../out/lstm"
    TRANSFORMER_RESULTS_PATH: str = "../out/transfomer"
    SAVED_MODELS_PATH: str = "../out/models"
    RUNS_LOG_PATH: str = "../runs"
