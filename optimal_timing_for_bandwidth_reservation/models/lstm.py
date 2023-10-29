import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A Long-Short Term Memory (LSTM) neural network module for predicting prices of multiple providers
    based on timestamp inputs.
    """

    def __init__(self, input_size=1, hidden_layer_size=100, num_providers=1):
        """
        Initializes the LSTM module.

        Args:
            input_size (int): The number of expected features in the input. Default is 1 (for timestamps).
            hidden_layer_size (int): The number of features in the hidden state of the LSTM cell.
            num_providers (int): The number of providers for which prices are predicted.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, num_providers)

    def forward(self, input_seq):
        """
        Performs a forward pass through the LSTM network.

        Args:
            input_seq (torch.Tensor): The input sequence to the network of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: The predicted output sequence of shape (batch_size, num_providers).
        """

        h0 = (
            torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(
                input_seq.device
            ),
            torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(
                input_seq.device
            ),
        )

        lstm_out, h0 = self.lstm(input_seq, h0)

        predictions = self.linear(lstm_out)

        return predictions
