"""
This script trains an LSTM and a Transformer model to predict future values of a time series.

Dependencies:
- torch.nn (nn module of PyTorch)
- torch.optim (optim module of PyTorch)
- models.transformer (TransformerModel class from models/transformer.py)
- models.lstm (LSTM class from models/lstm.py)
- config.params (Params dataclass from config/params.py)
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
from config.params import Params
from utils.data_processor import DataProcessor
from utils.batch_generator import BatchGenerator
from trainers.trainer import Trainer

params = Params()


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


def train_and_test(
    model, optimizer, loss_fn, device, scheduler=None, is_transformer=False
):
    trainer = Trainer(model, loss_fn, optimizer, device, scheduler, is_transformer)
    losses = []
    accuracies = []
    times = []

    for epoch in range(params.EPOCHS):
        start_time = time.time()
        batch_gen = BatchGenerator(train_inout_seq, params.BATCH_SIZE)

        if scheduler:
            loss, accuracy = trainer.train_with_scheduler(batch_gen)
        else:
            loss, accuracy = trainer.train(batch_gen)

        end_time = time.time()

        losses.append(loss)
        accuracies.append(accuracy)
        times.append(end_time - start_time)

        if epoch % params.FREQ_PRINTING == 0:
            writer.add_scalar(f"Loss/{type(model).__name__}", loss, epoch)
            writer.add_scalar(f"Accuracy/{type(model).__name__}", accuracy, epoch)
            print(
                f"{type(model).__name__} - epoch: {epoch:3} loss: {loss:10.8f} accuracy: {accuracy:5.2f}"
            )

    return losses, accuracies, times


if __name__ == "__main__":
    writer = SummaryWriter(f"{params.RUNS_LOG_PATH}/time_series_experiment")
    files = params.DATAFILES
    dp = DataProcessor(files)
    dp.load_data()
    train_inout_seq = dp.get_train_sequences()
    validation_inout_seq = dp.get_validation_sequences()
    test_seqs = dp.get_test_sequences()

    lstm_model = LSTM(num_providers=dp.num_providers).to(params.DEVICE)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    lstm_loss = nn.MSELoss()

    lstm_losses, lstm_accuracies, lstm_times = train_and_test(
        lstm_model, lstm_optimizer, lstm_loss, params.DEVICE
    )

    lstm_val_mse = evaluate_model(lstm_model, validation_inout_seq)
    print(f"LSTM MSE on validation data: {lstm_val_mse}")
    lstm_mse = evaluate_model(lstm_model, test_seqs)
    print(f"LSTM MSE on test data: {lstm_mse}")

    torch.save(lstm_model.state_dict(), f"{params.SAVED_MODELS_PATH}/lstm_model.pth")

    transformer_model = TransformerModel(
        ninp=1,
        nhead=params.N_HEAD,
        nhid=params.N_HIDDEN_L,
        nlayers=params.N_LAYERS,
        dropout=params.DROPOUT,
    ).to(params.DEVICE)
    transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=5e-3)
    transformer_loss = nn.MSELoss()
    transformer_scheduler = optim.lr_scheduler.StepLR(
        transformer_optimizer, 1.0, gamma=0.95
    )

    transformer_losses, transformer_accuracies, transformer_times = train_and_test(
        transformer_model,
        transformer_optimizer,
        transformer_loss,
        params.DEVICE,
        transformer_scheduler,
        is_transformer=True,
    )

    transformer_val_mse = evaluate_model(transformer_model, validation_inout_seq)
    print(f"Transformer MSE on validation data: {transformer_val_mse}")
    transformer_mse = evaluate_model(transformer_model, test_seqs)
    print(f"Transformer MSE on test data: {transformer_mse}")

    torch.save(
        transformer_model.state_dict(),
        f"{params.SAVED_MODELS_PATH}/transformer_model.pth",
    )

    np.save(f"{params.LSTM_RESULTS_PATH}/lstm_iil.npy", lstm_losses)
    np.save(f"{params.LSTM_RESULTS_PATH}/lstm_acc.npy", lstm_accuracies)
    np.save(f"{params.LSTM_RESULTS_PATH}/lstm_times.npy", lstm_times)
    np.save(f"{params.LSTM_RESULTS_PATH}/lstm_validation_iil.npy", lstm_val_mse)

    np.save(f"{params.TRANSFORMER_RESULTS_PATH}/tr_iil.npy", transformer_losses)
    np.save(f"{params.TRANSFORMER_RESULTS_PATH}/tr_acc.npy", transformer_accuracies)
    np.save(f"{params.TRANSFORMER_RESULTS_PATH}/tr_times.npy", transformer_times)
    np.save(
        f"{params.TRANSFORMER_RESULTS_PATH}tr_validation_iil.npy", transformer_val_mse
    )

    writer.close()
