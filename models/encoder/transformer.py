from typing import Tuple, Optional
from torch import nn
from torch import Tensor

from torch.nn.modules.transformer import _get_activation_fn, F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn import Linear, LayerNorm, Dropout

from models.attention.rel_self_attention import Rel_MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
    """
    __constants__ = ['batch_first']

    def __init__(
            self, 
            d_model: int, 
            nhead: int, 
            type_att: str = "self_att",  # 'self_att' or 'rel_self_att'
            dim_feedforward: int = 2048, 
            dropout: float = 0.1, 
            activation: str = "relu",
            layer_norm_eps: float = 1e-5,
            batch_first: bool = True,
            device: str = None,
            dtype: str = None,
            rezero: bool = False
        ) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(TransformerEncoderLayer, self).__init__()
        
        if type_att == 'self_att':
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
            self.use_rel_attn = False

        elif type_att == 'rel_self_att':
            self.self_attn = Rel_MultiheadAttention(d_model, nhead, dropout= dropout)
            self.use_rel_attn = True

        else:
            raise ValueError("The current version only supports 'self_att' or 'rel_self_att' attention type !")
        
        self.type_att = type_att
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.resweight = nn.Parameter(Tensor([0]))
        self.rezero = rezero

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if not self.use_rel_attn:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        else:
            src2 = self.self_attn(src)

        if self.rezero:
            src2 = src2 * self.resweight
            src = src + self.dropout1(src2)
        else:
            src = src + self.dropout1(src2)
            src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        if self.rezero:
            src2 = src2 * self.resweight
            src = src + self.dropout2(src2)
        else:
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src



class Encoder(nn.Module):

    """ Transformer Encoder

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
            type_att: str = 'self_att',
            pre_norm: bool = False,
            rezero: bool = False,
            *args, **kwargs
        ):

        super(Encoder, self).__init__()

        en_layer = TransformerEncoderLayer(d_model= d_model, nhead= n_head, batch_first= True, 
                            dropout= dropout, type_att= type_att, rezero= rezero)
        self.enc = nn.TransformerEncoder(en_layer, num_layers= n_layers)
        self.pre_norm = nn.LayerNorm(d_model)
        self.use_pre_norm = pre_norm
    

    def forward(self, 
            feats: Tensor
        ) -> Tuple[None, Tensor]:

        """Transformer encoder forward

        Args:
            feats: 3D torch Tensor (B x S x D)
        
        Returns:
            feats: 3D torch Tensor (B x S x D)

        """

        if self.use_pre_norm:
            feats = self.pre_norm(feats)

        feats = self.enc(feats)

        return None, feats


class InterEncoder(nn.Module):

    """ Transformer InterEncoder

    Args:

        n_layers: the number of encoder layer
        d_model: input features dimension
        n_head: transformer attention head
        dropout: transformer dropout
        num_intermediate_CTC_layer: the number of layer to compute intermediate CTC

    """

    def __init__(self,
            n_layers: int = 12, 
            d_model: int = 512, 
            n_head: int = 8, 
            dropout: float = 0.3, 
            type_att: str = 'self_att',
            num_intermediate_CTC_layer: int = 6
        ):

        super(InterEncoder, self).__init__()

        n_continue_layers = n_layers - num_intermediate_CTC_layer

        assert num_intermediate_CTC_layer >= 6, "num_intermediate_CTC_layer must be larger than 6"
        assert n_continue_layers >= 6, "(n_layers - num_intermediate_CTC_layer) must be greater than or equal to 6 !"

        en_layer = TransformerEncoderLayer(d_model= d_model, nhead= n_head, batch_first= True, dropout= dropout, type_att= type_att)

        self.inter_block =  nn.TransformerEncoder(en_layer, num_layers= num_intermediate_CTC_layer)
        self.continue_layers = nn.TransformerEncoder(en_layer, num_layers= n_continue_layers)
    

    def forward(self, 
            feats: Tensor
        ) -> Tuple[Tensor, Tensor]:

        """ Transformer InterEncoder Forward

        Args:
            feats: 3D torch Tensor (B x S x D)
        
        Returns:
            feats: 3D torch Tensor (B x S x D)

        """

        inter_out = self.inter_block(feats)
        out = self.continue_layers(inter_out)

        return inter_out, out