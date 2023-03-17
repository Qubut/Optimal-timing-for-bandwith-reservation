import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    A transformer-based neural network model for sequence to scalar regression.

    Args:
        ninp (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        nhid (int): The dimension of the feedforward network model.
        nlayers (int): The number of nn.TransformerEncoderLayer in the nn.TransformerEncoder.
        dropout (float): The dropout probability.

    Attributes:
        model_type (str): The type of the model, set to "Transformer".
        pos_encoder (_PositionalEncoding): The positional encoding layer.
        transformer_encoder (nn.TransformerEncoder): The transformer encoder layer.
        decoder (nn.Linear): The linear layer that maps the transformer output to a scalar value.

    Methods:
        generate_square_subsequent_mask(sz): Generates a square mask for the sequence.
        init_weights(): Initializes the weights of the linear layer.
        forward(src, src_mask): The forward pass of the model.

    """

    def __init__(self, ninp: int, nhead: int, nhid: int, nlayers: int, dropout=0.5):
        super(TransformerModel, self).__init__()

        # Import necessary modules
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        # Set model attributes
        self.model_type = 'Transformer'
        self.pos_encoder = _PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, 1)

        # Initialize weights
        self.init_weights()

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence.

        Args:
            sz (int): The length of the sequence.

        Returns:
            torch.Tensor: The mask tensor.

        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        """
        Initializes the weights of the linear layer.

        """
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Args:
            src (torch.Tensor): The input sequence tensor of shape (seq_len, batch_size, ninp).
            src_mask (torch.Tensor): The mask tensor of shape (seq_len, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1).

        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output[-1]
        output = self.decoder(output)
        return output


class _PositionalEncoding(nn.Module):
    """
    Private class for the positional encoding layer of the transformer model.

    Args:
        d_model (int): The number of expected features in the input.
        dropout (float): The dropout probability.
        max_len (int): The maximum length of the input sequence.

    Attributes:
        dropout (nn.Dropout): The dropout layer.
        _pe (torch.Tensor): The positional encoding tensor.

    Methods:
        forward(x): The forward pass of the layer.

    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(_PositionalEncoding, self).__init__()

        # Initialize the dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Generate the positional encoding tensor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register the positional encoding tensor as a buffer
        self.register_buffer('_pe', pe)

    def forward(self, x):
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with positional encoding applied.
        """
        # Apply the positional encoding tensor to the input tensor
        x = x + self._pe[:x.size(0), :]

        # Apply dropout to the output tensor
        return self.dropout(x)
