"""
This script trains an LSTM and a Transformer model to predict future values of a time series.

Dependencies:
- torch.nn (nn module of PyTorch)
- torch.optim (optim module of PyTorch)
- models.transformer (TransformerModel class from models/transformer.py)
- models.lstm (LSTM class from models/lstm.py)
- config.config (bsize, device, delta, epochs, freq_printing constants from config/config.py)
- utils.data_processor (DataProcessor class from utils/data_processor.py)
- utils.batch_generator (BatchGenerator class from utils/batch_generator.py)
- trainers.trainer (Trainer class from trainers/trainer.py)
"""

import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerModel
from models.lstm import LSTM
from config.config import bsize, device, delta, epochs, freq_printing
from utils.data_processor import DataProcessor
from utils.batch_generator import BatchGenerator
from trainers.trainer import Trainer

if __name__ == "__main__":
    # Creating DataProcessor object to load and process the data
    dp = DataProcessor("train.csv", "test.csv", delta, device)
    dp.load_data()
    train_inout_seq = dp.create_inout_sequences()

    # Creating LSTM model and trainer
    lstm_model = LSTM().to(device)
    lstm_loss = nn.MSELoss()
    lstm_opt = optim.Adam(lstm_model.parameters(), lr=1e-3)
    lstm_trainer = Trainer(lstm_model, lstm_loss, lstm_opt)

    # Training LSTM model
    for i in range(epochs):
        batch_gen = BatchGenerator(train_inout_seq, bsize)
        loss = lstm_trainer.train(batch_gen)
        if i % freq_printing == 0:
            print(f"epoch: {i:3} loss: {loss:10.8f}")

    # Creating Transformer model and trainer
    transformer_model = TransformerModel(100, 10, 10, 1, 0.2).to(device)
    transformer_loss = nn.MSELoss()
    transformer_opt = optim.AdamW(transformer_model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(transformer_opt, 1.0, gamma=0.95)
    transformer_trainer = Trainer(
        transformer_model, transformer_loss, transformer_opt, scheduler
    )

    # Training Transformer model
    for i in range(epochs):
        batch_gen = BatchGenerator(train_inout_seq, bsize)
        loss = transformer_trainer.train_with_scheduler(batch_gen)
        if i % freq_printing == 0:
            print(f"epoch: {i:3} loss: {loss:10.8f}")
