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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

from models.transformer import TransformerModel
from models.lstm import LSTM
from config.config import bsize, device, delta, epochs, freq_printing
from utils.data_processor import DataProcessor
from utils.batch_generator import BatchGenerator
from trainers.trainer import Trainer


def train_and_test(model, optimizer, loss_fn, device, scheduler=None):
    trainer = Trainer(model, loss_fn, optimizer, device, scheduler)
    losses = []
    times = []

    for epoch in range(epochs):
        start_time = time.time()
        batch_gen = BatchGenerator(train_inout_seq, bsize)

        if scheduler:
            loss = trainer.train_with_scheduler(batch_gen)
        else:
            loss = trainer.train(batch_gen)

        end_time = time.time()

        losses.append(loss)
        times.append(end_time - start_time)

        if epoch % freq_printing == 0:
            writer.add_scalar(f"Loss/{type(model).__name__}", loss, epoch)
            print(f"{type(model).__name__} - epoch: {epoch:3} loss: {loss:10.8f}")

    return losses, times


def evaluate_model(model, test_seqs):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input_seq, target_seq in test_seqs:
            output_seq = model(input_seq)
            y_true.extend(target_seq.cpu().numpy())
            y_pred.extend(output_seq.cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    return mse


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/time_series_experiment")
    dp = DataProcessor("./datasets/Dataset_NO1.csv", delta, device)
    dp.load_data()
    train_inout_seq = dp.get_train_sequences()
    test_seqs = dp.get_test_sequences()

    # LSTM model training and testing
    lstm_model = LSTM().to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    lstm_loss = nn.MSELoss()

    lstm_losses, lstm_times = train_and_test(
        lstm_model, lstm_optimizer, lstm_loss, device
    )
    lstm_mse = evaluate_model(lstm_model, test_seqs)
    print(f"LSTM MSE on test data: {lstm_mse}")

    # Transformer model training and testing
    transformer_model = TransformerModel(100, 10, 10, 1, 0.2).to(device)
    transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=5e-3)
    transformer_loss = nn.MSELoss()
    transformer_scheduler = optim.lr_scheduler.StepLR(
        transformer_optimizer, 1.0, gamma=0.95
    )

    transformer_losses, transformer_times = train_and_test(
        transformer_model,
        transformer_optimizer,
        transformer_loss,
        device,
        transformer_scheduler,
    )
    transformer_mse = evaluate_model(transformer_model, test_seqs)
    print(f"Transformer MSE on test data: {transformer_mse}")

    # Save the losses and times
    np.save("out/lstm/lstm_iil.npy", lstm_losses)
    np.save("out/transformer/tr_iil.npy", transformer_losses)
    np.save("out/lstm/lstm_times.npy", lstm_times)
    np.save("out/transformer/tr_times.npy", transformer_times)

    writer.close()
