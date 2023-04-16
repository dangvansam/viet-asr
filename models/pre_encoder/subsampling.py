import math

from torch import nn
from torch import Tensor


class Conv2dSubsampling_2(nn.Module):

    """
    Convolutional 2D subsampling (to 1/2 length)
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - inputs (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - outputs (batch, time, dim): Tensor produced by the convolution
    
    Notes:
        - L_sub = (L_mel - 5)/2
    """

    def __init__(self, 
            input_dim: int = 80,
            in_channels: int = 1, 
            out_channels: int = 256,
            out_dim: int = 512,
            input_dropout_p: float = 0.1
        ) -> None:

        super(Conv2dSubsampling_2, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        sumsampled_dim = math.floor( (input_dim -3)/2 + 1 )
        sumsampled_dim = math.floor( (sumsampled_dim - 3)/1 + 1 )

        self.input_projection = nn.Sequential(
            nn.Linear(out_channels * sumsampled_dim, 2 * out_dim),
            nn.Dropout(p=input_dropout_p),
        )

    def forward(self, 
            inputs: Tensor
        ) -> Tensor:

        inputs = inputs.permute(0,2,1)
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, sumsampled_dim, subsampled_lengths = outputs.size()

        outputs = outputs.permute(0, 3, 1, 2)

        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        outputs = self.input_projection(outputs)

        return outputs


class Conv2dSubsampling_4(nn.Module):

    """
    Convolutional 2D subsampling (to 1/4 length)
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - inputs (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - outputs (batch, time, dim): Tensor produced by the convolution
    """

    def __init__(self, 
            input_dim: int = 80,
            in_channels: int = 1, 
            out_channels: int = 256,
            out_dim: int = 512,
            input_dropout_p: float = 0.1
        ) -> None:

        super(Conv2dSubsampling_4, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        sumsampled_dim = math.floor( (input_dim -3)/2 + 1 )
        sumsampled_dim = math.floor( (sumsampled_dim - 3)/2 + 1 )

        self.input_projection = nn.Sequential(
            nn.Linear(out_channels * sumsampled_dim, out_dim),
            nn.Dropout(p=input_dropout_p),
        )

    def forward(self, 
            inputs: Tensor
        ) -> Tensor:

        inputs = inputs.permute(0,2,1)
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, sumsampled_dim, subsampled_lengths = outputs.size()

        outputs = outputs.permute(0, 3, 1, 2)

        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        outputs = self.input_projection(outputs)

        return outputs

class Conv2dSubsampling_4_Linear(nn.Module):

    """
    Convolutional 2D subsampling (to 1/4 length)
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - inputs (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - outputs (batch, time, dim): Tensor produced by the convolution
    """

    def __init__(self, 
            input_dim: int = 80,
            in_channels: int = 1, 
            out_channels: int = 256,
            out_dim: int = 512,
            input_dropout_p: float = 0.1
        ) -> None:

        super(Conv2dSubsampling_4_Linear, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
        )

        sumsampled_dim = math.floor( (input_dim -3)/2 + 1 )
        sumsampled_dim = math.floor( (sumsampled_dim - 3)/2 + 1 )

        self.input_projection = nn.Sequential(
            nn.Linear(out_channels * sumsampled_dim, out_dim),
        )

    def forward(self, 
            inputs: Tensor
        ) -> Tensor:

        inputs = inputs.permute(0,2,1)
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, sumsampled_dim, subsampled_lengths = outputs.size()

        outputs = outputs.permute(0, 3, 1, 2)

        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        outputs = self.input_projection(outputs)

        return outputs

class Conv2dSubsampling_6(nn.Module):

    """
    Convolutional 2D subsampling (to 1/6 length)
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - inputs (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - outputs (batch, time, dim): Tensor produced by the convolution
    """

    def __init__(self, 
            input_dim: int = 80,
            in_channels: int = 1, 
            out_channels: int = 256,
            out_dim: int = 512,
            input_dropout_p: float = 0.1
        ) -> None:

        super(Conv2dSubsampling_6, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=3),
            nn.ReLU(inplace= True),
        )

        sumsampled_dim = math.floor( (input_dim - 3)/2 + 1 )
        sumsampled_dim = math.floor( (sumsampled_dim - 5)/3 + 1 )

        self.input_projection = nn.Sequential(
            nn.Linear(out_channels * sumsampled_dim, out_dim),
            nn.Dropout(p=input_dropout_p),
        )


    def forward(self, 
            inputs: Tensor
        ) -> Tensor:

        inputs = inputs.permute(0,2,1)
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, sumsampled_dim, subsampled_lengths = outputs.size()

        outputs = outputs.permute(0, 3, 1, 2)

        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        # print(outputs.shape)
        outputs = self.input_projection(outputs)

        return outputs