from torch import nn
from torch import Tensor


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

