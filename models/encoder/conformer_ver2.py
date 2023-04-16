import copy
from typing import Tuple

from torch import nn
from torch import Tensor
from torch.nn import Dropout

from models.attention.rel_self_attention import Rel_MultiheadAttention
from utils import initialize_weights



class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        ffn_mult = 2,
        dropout = 0.1
    ):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.SiLU(inplace= True),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

        self.output_dropout = nn.Dropout(dropout)

        initialize_weights(self.net)


    def forward(
            self, 
            x: Tensor,
            resweight: nn.Parameter,
        ) -> Tensor:

        return self.output_dropout(resweight * self.net(x))


class ConvolutionModule(nn.Module):

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            dropout: float = 0.1,
            conv_mult: int = 2,
        ) -> None:

        super(ConvolutionModule, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels= dim, out_channels= dim * conv_mult, kernel_size= kernel_size, stride= 1, padding= "same"),
            nn.SiLU(inplace= True),
            nn.Conv1d(in_channels= dim * conv_mult, out_channels= dim, kernel_size= kernel_size, stride= 1, padding= "same"),
        )

        self.output_dropout = nn.Dropout(p= dropout)
        initialize_weights(self.net)

    def forward(
            self, 
            x: Tensor,
            resweight: nn.Parameter,
        ) -> Tensor:

        x = x.permute(0,2,1)
        x = self.output_dropout(resweight * self.net(x))
        x = x.permute(0,2,1)

        return x


class ConformerEncoderLayer(nn.Module):
    r"""Conformer Encoder Layer with Rezero technical
    """

    def __init__(
            self, 
            d_model: int, 
            nhead: int = 8, 
            dropout: float = 0.1, 
            conv_kernel_size: int = 3,
            conv_mult: int = 2,
            ffn_mult: int = 2
        ) -> None:

        super(ConformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.self_attn = Rel_MultiheadAttention(d_model, nhead)
        
        # Implementation of Feedforward model
        self.ffn1 = FeedForward(d_model, ffn_mult)
        self.ffn2 = FeedForward(d_model, ffn_mult)
        self.conv = ConvolutionModule(d_model, kernel_size= conv_kernel_size, dropout= dropout, conv_mult= conv_mult)

        self.dropout1 = Dropout(dropout)

        self.resweight = nn.Parameter(Tensor([0]))

        initialize_weights(self.self_attn)


    def forward(
            self, 
            src: Tensor, 
        ) -> Tensor:

        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).

        Outputs:

            same shape as src
        """

        src = src + self.ffn1(src, self.resweight)
        src = src + self.dropout1( self.resweight * self.self_attn(src) )
        src = src + self.conv(src, self.resweight)
        src = src + self.ffn2(src, self.resweight)

        return src


class Encoder(nn.Module):

    def __init__(
            self,
            n_layers: int = 6,
            pre_norm: bool = True,
            *args, 
            **kwargs
        ) -> None:

        super(Encoder, self).__init__()

        conformer_layer = ConformerEncoderLayer(*args, **kwargs)
        self.encs = self._get_clones(conformer_layer, n_layers)
        self.use_pre_norm = pre_norm

        self.pre_norm = nn.LayerNorm(conformer_layer.d_model)


    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

    def forward(self, x: Tensor) -> Tuple[None, Tensor]:

        if self.use_pre_norm:
            x = self.pre_norm(x)
        
        for con_layer in self.encs:
            x = con_layer(x)
        
        return None, x