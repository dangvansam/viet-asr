import copy
from typing import Callable, Optional, Tuple

from torch import nn
from torch import Tensor
import torch.nn.functional as F


def _get_clones(module: Callable, N: int) -> Callable:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BaseTrans(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(
            self, 
            decoder_layer: nn.Module, 
            num_layers: int, 
            d_model: int,
            d_feedforward: int,
            pre_norm: bool = False
        ):

        super(BaseTrans, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_feedforward = d_feedforward
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

        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        output = tgt

        if self.use_pre_norm:
            output = self.pre_norm(output)

        for mod in self.layers:
            output, attn_output_weights = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        return output, attn_output_weights

class TransformerDecoder(BaseTrans):
    """ Transformer Decoder
    Args:

        n_layers: the number of encoder layer
        d_model: input features dimension
        n_head: transformer attention head
        dropout: transformer dropout

    """
    def __init__(self,
            n_layers: int = 6, 
            d_model: int = 512,
            d_feedforward: int = 2048,
            n_head: int = 8, 
            dropout: float = 0.1,
            pre_norm: bool = False,
        ):
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            dim_feedforward=d_feedforward,
            nhead=n_head,
            dropout=dropout
        )
        super(TransformerDecoder, self).__init__(layer, n_layers, d_model, pre_norm)
