import math

from torch import nn
from torch import Tensor


class SubWord_Embedding(nn.Module):

    """SubWord embeeding layer

    Args:
        vocab_size: input features dimension
        embed_size: output features dimension

    """

    def __init__(self, 
            vocab_size: int, 
            embed_size: int, 
        ):

        super(SubWord_Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim= embed_size)
        self.factor = math.sqrt(embed_size)
    

    def forward(self, input_decoder: Tensor) -> Tensor:

        """
        Args:
            input_decoder: 2D tensor (B, SD)
        
        Returns:
            output_embedding: 3D tensor (B, SD, embed_size)
        """

        input_decoder = self.embedding(input_decoder) * self.factor

        return input_decoder