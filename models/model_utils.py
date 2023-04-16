from torch import nn
from torch import Tensor


class IC_layer(nn.Module):

    """IC_layer, the implementation of paper: https://arxiv.org/pdf/1905.05928.pdf

    The IC layer consistently outperforms the baseline approaches 
    with more stable training process, faster convergence speed 
    and better convergence limit on CIFAR10/100 and ILSVRC2012 datasets

    Args:
        dim_features (int): the number of expected features in the input (required).
        dropout_rate (float): dropout rate (required).

    """

    def __init__(self,
                dim_features: int,
                dropout_rate: float = 0.05):
        super(IC_layer, self).__init__()

        self.norm_layer = nn.BatchNorm1d(dim_features)
        self.dropout_layer = nn.Dropout(p= dropout_rate, inplace= True)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 3D torch Tensor with shape (B x D x S)

        Returns:
            y: 3D torch Tensor with shape (B x D x S)

        """
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)
        x = self.dropout_layer(x)
        x = x.permute(0, 2, 1)

        return x



class Reshape_Layer(nn.Module):
    """Reshape layer

    Reshape of torch Tensor, (B x S x D) -> (B x D x S)

    B: Batch size
    S: Sequence length
    D: Dimension of features

    """

    def __init__(self):
        super(Reshape_Layer, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 3D torch Tensor with shape (B x S x D)

        Returns:
            y: 3D torch Tensor with shape (B x D x S)

        """
        return x.permute(0, 2, 1)

