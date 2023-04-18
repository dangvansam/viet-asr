import math

import torch

from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):

    """ Positional Encoding with Concat Style or Add Style

    Args:
        d_model: dimension of features vector
        dropout: output dropout
        max_len: max position

    """

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 style: str = 'concat'
                 ) -> None:

        super(PositionalEncoding, self).__init__()

        self.use_concat = False
        self.dropout = nn.Dropout(p=dropout)

        if style == 'concat':
            self.use_concat = True
            self.linear = nn.Linear(
                in_features=2 * d_model, out_features=d_model)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: acoustic features 

        Returns:
            y: acoustic features plus with positional encoding vector

        Shapes:
            x: 3D torch Tensor (B x S x D)
            y: 3D torch Tensor (B x S x D)
        """

        if self.use_concat:
            B, S, D = x.shape
            pe = self.pe[:, :S]
            pe = pe.repeat(B, 1, 1)
            x = torch.cat([x, pe], dim=2)
            x = self.linear(x)
        else:
            x = x + self.pe[:, : x.size(1)]

        x = self.dropout(x)
        return x
