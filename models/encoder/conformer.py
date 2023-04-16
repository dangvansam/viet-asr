from typing import Tuple
from torch import nn
from torch import Tensor

from conformer import ConformerBlock




class BaseEnc(nn.Module):
    
    def __init__(self):
        super(BaseEnc, self).__init__()
    
    def build_block(
            self,
            n_layers: int,
            dim: int, 
            dim_head: int, 
            heads: int, 
            attn_dropout: float, 
            ff_dropout: float, 
            conv_dropout: float,
            rezero: bool
        ) -> nn.Sequential: 

        block = list()

        for _ in range(n_layers):

            conformer_block = ConformerBlock(
                            dim= dim, 
                            dim_head= dim_head, 
                            heads= heads, 
                            attn_dropout= attn_dropout, 
                            ff_dropout= ff_dropout, 
                            conv_dropout= conv_dropout,
                            rezero= rezero
                        )

            block.append(conformer_block)
        
        return nn.Sequential(*block)


class Encoder(BaseEnc):

    """ Conformer Encoder

    Args:

        d_model: input features dimension 
        dim_conformer_att_head: conformer attention dimension
        conformer_head: conformer attention head
        attn_dropout: conformer attention dropout
        ff_dropout: conformer feed forward dropout
        conv_dropout: conformer conv dropout

    """

    def __init__(
            self,
            n_layers: int = 12,
            d_model: int = 256, 
            dim_conformer_att_head: int = 256, 
            conformer_head: int = 4, 
            attn_dropout: float = 0.3, 
            ff_dropout: float = 0.3, 
            conv_dropout: float = 0.3,
            rezero: bool = False 
        ):

        super(Encoder, self).__init__()

        self.enc = self.build_block(n_layers, d_model, dim_conformer_att_head, conformer_head,
                                            attn_dropout, ff_dropout, conv_dropout, rezero)

    def forward(self, 
            feats: Tensor
        ) -> Tuple[None, Tensor]:

        """Conformer Encoder Forward

        Args:
            feats: 3D torch Tensor (B x S x D)
        
        Returns:
            feats: 3D torch Tensor (B x S x D)

        """

        feats = self.enc(feats)

        return None, feats


class InterEncoder(BaseEnc):

    """ Conformer InterEncoder

    Args:

        d_model: input features dimension 
        dim_conformer_att_head: conformer attention dimension
        conformer_head: conformer attention head
        attn_dropout: conformer attention dropout
        ff_dropout: conformer feed forward dropout
        conv_dropout: conformer conv dropout
        num_intermediate_CTC_layer: the number of layer to compute intermediate CTC

    """

    def __init__(
            self,
            n_layers: int = 12,
            d_model: int = 256, 
            dim_conformer_att_head: int = 256, 
            conformer_head: int = 4, 
            attn_dropout: float = 0.3, 
            ff_dropout: float = 0.3, 
            conv_dropout: float = 0.3,
            num_intermediate_CTC_layer: int = 6,
        ):

        super(InterEncoder, self).__init__()

        n_continue_layers = n_layers - num_intermediate_CTC_layer

        assert num_intermediate_CTC_layer >= 6, "num_intermediate_CTC_layer must be larger than 6"
        assert n_continue_layers >= 6, "(n_layers - num_intermediate_CTC_layer) must be greater than or equal to 6 !"

        self.inter_block = self.build_block(num_intermediate_CTC_layer, d_model, dim_conformer_att_head, conformer_head,
                                            attn_dropout, ff_dropout, conv_dropout)

        self.continue_block = self.build_block(n_continue_layers, d_model, dim_conformer_att_head, conformer_head,
                                            attn_dropout, ff_dropout, conv_dropout)


    def forward(self, 
            feats: Tensor
        ) -> Tuple[Tensor, Tensor]:

        """ Conformer InterEncoder Forward

        Args:
            feats: 3D torch Tensor (B x S x D)
        
        Returns:
            feats: 3D torch Tensor (B x S x D)

        """

        out_inter = self.inter_block(feats)
        out = self.continue_block(out_inter)

        return out_inter, out