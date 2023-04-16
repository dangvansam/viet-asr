from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor

from models.seq2seq import ASR
from utils import initialize_weights


class Encoder(nn.Module):

    r""" Ensemble SLL Encoder
    """
    
    def __init__(
            self,
            vocab_size: int,
            d_model_list: list,
            ssl_models: nn.ModuleList,
            *args, **kwargs
        ) -> None:

        super(Encoder, self).__init__()

        self.ssl_models = ssl_models
        
        for model in self.ssl_models:
            model.encoder.requires_grad_(False)

        self.alpha = nn.Parameter(Tensor([0.5]), requires_grad= True)
        self.beta = nn.Parameter(Tensor([0.5]), requires_grad= True)

    def forward(
            self, 
            waveforms: Tensor, 
        ) -> Tuple[None, Tensor]:
        
        r"""Pass the input through the encoder layer.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.

        Outputs:

            Tensor of shape: `(batch, time frame, feature dimension)`
        """
        
        x = list()
        y = None

        for model in self.ssl_models:
            feat = model.run_encoder(waveforms)
            feat = model.encoder_final_fc(feat)
            x.append(feat)

        x = self.alpha * x[0] + self.beta * x[1]
        
        return x, y

    def run_encoder(self, waveforms: Tensor) -> Tensor:
        x = list()
        
        for model in self.ssl_models:
            feat = model.run_encoder(waveforms)
            feat = model.encoder_final_fc(feat)
            x.append(feat)

        x = self.alpha * x[0] + self.beta * x[1]
        return x
