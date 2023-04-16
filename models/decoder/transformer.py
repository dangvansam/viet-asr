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
            pre_norm: bool = False
        ):

        super(BaseTrans, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
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


class RZTXDecoderLayer(nn.Module):

    r"""RZTXDecoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> decoder_layer = RZTXDecoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = decoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first= True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first= batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first= batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(Tensor([0]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) * self.resweight
        tgt2, attn_output_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2) * self.resweight

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2) * self.resweight

        return tgt, attn_output_weights


class Decoder(BaseTrans):

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
            n_head: int = 8, 
            dropout: float = 0.1,
            rezero: bool = False,
            pre_norm: bool = False,
        ):

        if rezero:
            layer = RZTXDecoderLayer(d_model= d_model, nhead= n_head, batch_first= True, dropout= dropout)
        else:
            layer = nn.TransformerDecoderLayer(d_model= d_model, nhead= n_head, batch_first= True, dropout= dropout)

        super(Decoder, self).__init__(layer, n_layers, d_model, pre_norm)
