import random
from typing import Tuple

import torch
from torch import nn
from torch import Tensor


from execution_time import ExecutionTime
extime = ExecutionTime(console= False)


@extime.timeit
def make_mask(
        inputs: Tensor,
        mask_size: int = 5,
        mask_ratio: float = 0.10
    ) -> Tuple[Tensor, Tensor]:

    B, S, _ = inputs.shape

    mask = torch.zeros(inputs.shape, dtype=torch.bool)
    mask_input = torch.stack([torch.zeros(S)] * B, dim=0)
    mask_input = mask_input.long()
    
    max_number_mask = int(mask_ratio * S)
    # n_mask = random.randint(0, max_number_mask)

    indexs = list(range(S - mask_size))
    random.shuffle(indexs)
    mask_indexs = indexs[:max_number_mask]

    for i in mask_indexs:
        mask[:, i : i+mask_size, :] = True

    return mask, mask_input


class Mask_Hidden_Features(nn.Module):

    """
    Mask acoutics features after subsampling layer
    
    Args:
        mask_size (int): The number of time steps to be masked
        mask_dim (int): The hidden feature dimension of the mask

    Inputs: inputs
        - inputs (batch, time, dim): Tensor containing sequence of inputs
        - mask (batch, time, dim): The boolean tensor mask
        - mask_input (batch, time): The input mask embedding

    Returns: outputs, output_lengths
        - outputs (batch, time, dim): Masked features tensor
    """

    def __init__(self, 
            mask_dim: int = 512,
        ) -> None:

        super(Mask_Hidden_Features, self).__init__()

        self.mask_emb = nn.Embedding(num_embeddings= 1, embedding_dim= mask_dim)

    def forward(self, 
            inputs: Tensor,
            mask: torch.BoolTensor = None,
            mask_input: Tensor = None
        ) -> Tensor:

        mask_embedd = self.mask_emb(mask_input)
        output = (~mask).long() * inputs + mask.long() * mask_embedd

        return output