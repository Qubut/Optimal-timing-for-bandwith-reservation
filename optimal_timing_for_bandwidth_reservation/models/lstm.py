import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A Long-Short Term Memory (LSTM) neural network module for time series prediction.
    """

    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        """
        Initializes the LSTM module.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_layer_size (int): The number of features in the hidden state of the LSTM cell.
            output_size (int): The number of output features.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Define the linear layer to produce the output
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """
        Performs a forward pass through the LSTM network.

        Args:
            input_seq (torch.Tensor): The input sequence to the network of shape (seq_len, batch_size, input_size).

        Returns:
            torch.Tensor: The predicted output sequence of shape (batch_size, output_size).
        """
        # Initialize the hidden and cell states to zeros
        h0 = (
            torch.zeros(1, input_seq.size(1), self.hidden_layer_size).to(
                input_seq.device
            ),
            torch.zeros(1, input_seq.size(1), self.hidden_layer_size).to(
                input_seq.device
            ),
        )

        # Pass the input sequence through the LSTM layer
        lstm_out, h0 = self.lstm(input_seq, h0)

        # Pass the final LSTM output through the linear layer to produce the prediction
        predictions = self.linear(lstm_out[-1])

        return predictions
