import copy
from typing import Callable, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn


def _get_clones(module: Callable, N: int) -> Callable:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder is a stack of N decoder layers
    """
    __constants__ = ['norm']

    def __init__(
            self, 
            n_layers: int = 6, 
            n_head: int = 4, 
            dropout: float = 0.1,
            d_model: int = 256,
            d_feedforward: int = 1024,
            pre_norm: bool = False
        ):

        super(TransformerDecoder, self).__init__()
        
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            dim_feedforward=d_feedforward,
            nhead=n_head,
            dropout=dropout
        )
        self.layers = _get_clones(layer, n_layers)
        self.pre_norm = nn.LayerNorm(d_model)
        self.use_pre_norm = pre_norm

    def forward(self, 
            tgt: Tensor, 
            memory: Tensor, 
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None, 
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:

        """
        Pass the inputs (and mask) through the decoder layer in turn.
        """

        output = tgt

        if self.use_pre_norm:
            output = self.pre_norm(output)
        
        output = output.transpose(0, 1)
        memory = memory.transpose(0, 1)
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_key_padding_mask=tgt_mask,
                memory_key_padding_mask=memory_mask
            )
        output = output.transpose(0, 1)
        return output