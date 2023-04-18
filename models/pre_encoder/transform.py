import math
import numpy as np

from typing import Callable, Optional

import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB



class RangeNorm(torch.nn.Module):
    r"""Normalize spectrogram to zero mean and unit variance
    """

    __constants__ = ['padding_value_in_db']

    def __init__(self, padding_value_in_db: float = -100.0) -> None:
        super(RangeNorm, self).__init__()

        self.padding_value_in_db = padding_value_in_db


    def forward(
            self, 
            spectrogram: Tensor,
        ) -> Tensor:

        """
        Args:
            spectrogram (Tensor): Tensor shape of `(..., freq, time)`.

        Returns:
            Tensor: norm of the input tensor, shape of `(..., freq, time)`.
        """
        
        B, D, T = spectrogram.shape
        mask = (spectrogram != self.padding_value_in_db)
        n = mask.sum(dim=(1,2))
        mask = mask.long().to(spectrogram.dtype)

        # compute mean
        spectrogram_mean = (mask * spectrogram).sum(dim=(1,2))/n
        spectrogram_mean = spectrogram_mean.unsqueeze(1).repeat((1, D * T)).reshape((B, D, T))
        # compute std
        spectrogram_std = torch.sqrt(((spectrogram-spectrogram_mean)**2 * mask).sum(dim=(1,2))/(n-1))
        spectrogram_std = spectrogram_std.unsqueeze(1).repeat((1, D * T)).reshape((B, D, T))

        norm_spec = mask * (spectrogram - spectrogram_mean) / spectrogram_std
        return norm_spec


class SpecAugt(torch.nn.Module):
    r"""Mask Mel-Spectrogram by x and y-axis
    """

    __constants__ = ['padding_value']

    def __init__(
            self, 
            prob: float = 0.5,
            num_freq_mask: int = 1,
            num_time_mask: int = 2,
            max_freq_mask_frame: int = 30,
            max_time_mask_frame: int = 30,
            max_time_swap_frame: int = 30,
            padding_value: float = -0.0,
        ) -> None:
        
        '''
        Args:
            x: 2D tensor features input (freq, time)
            num_freq_mask: the number of time apply freq mask
            num_tim_mask: the number of time apply time mask
            max_freq_mask_frame: the maximum number of frame when apply freq mask
            max_time_mask_frame: the maximum number of frame when apply time mask
        
        Returns:
            x: 2D masked tensor (freq, time)
        '''

        super(SpecAugt, self).__init__()

        self.prob = prob
        self.padding_value = padding_value
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
        self.max_freq_mask_frame = max_freq_mask_frame
        self.max_time_mask_frame = max_time_mask_frame
        self.max_time_swap_frame = max_time_swap_frame
    
    def augment(
            self,
            x: Tensor,
            seq_length: int,
        ) -> Tensor:

        dim, _ = x.shape
            
        for i in range(self.num_time_mask):
            # time masking
            if np.random.uniform(0,1) < self.prob:
                delta_t = np.random.randint(0, self.max_time_mask_frame)
                if seq_length < self.max_time_mask_frame: continue
                # print("time mask: {}".format(delta_t))
                t0 = np.random.randint(0, seq_length - delta_t)
                x[:, t0 : t0 + delta_t] = 0
        
        for i in range(self.num_freq_mask):
            # freq masking
            if np.random.uniform(0,1) < self.prob:
                delta_f = np.random.randint(0, self.max_freq_mask_frame)
                # print("freq mask: {}".format(delta_f))
                f0 = np.random.randint(0, dim - delta_f)
                x[f0 : f0 + delta_f, :] = 0
        
        if np.random.uniform(0,1) < self.prob:
            # time warping
            delta_t = np.random.randint(0, self.max_time_swap_frame)

            t0 = np.random.randint(0, seq_length - delta_t)
            temp0 = torch.clone(x[:, t0 : t0 + delta_t])

            t1 = np.random.randint(0, seq_length - delta_t)
            temp1 = torch.clone(x[:, t1 : t1 + delta_t])

            x[:, t0 : t0 + delta_t] = temp1
            x[:, t1 : t1 + delta_t] = temp0
        
        return x


    def forward(
            self, 
            spectrogram: Tensor,
        ) -> Tensor:

        """
        Args:
            spectrogram (Tensor): Tensor shape of `(..., freq, time)`.

        Returns:
            Tensor: norm of the input tensor, shape of `(..., freq, time)`.
        """
        
        B, D, T = spectrogram.shape

        for i in range(B):
            spec_i = spectrogram[i]
            seq_length = (spec_i[0]==0).nonzero().tolist()

            if not seq_length:
                seq_length = T
            else:
                seq_length = seq_length[0][0]

            spectrogram[i] = self.augment(spec_i, seq_length)

        return spectrogram


class CMVN(torch.nn.Module):
    r"""Normalize spectrogram to mean values
    """

    def __init__(self, 
            infor: Tensor,
            global_norm: bool = True
        ) -> None:

        super(CMVN, self).__init__()
        
        if global_norm:
            self.global_mean_vector = torch.Tensor(infor['global_mean_value'])
        self.global_norm = global_norm

    def forward(
            self, 
            spectrogram: Tensor,
        ) -> Tensor:

        """
        Args:
            spectrogram (Tensor): Tensor shape of `(..., freq, time)`.

        Returns:
            Tensor: norm of the input tensor, shape of `(..., freq, time)`.
        """
        
        B, D, T = spectrogram.shape
        spectrogram_mean = spectrogram.mean(dim=2).reshape((B, D, 1)).repeat((1, 1, T))

        if self.global_norm:
            global_mean = self.global_mean_vector.unsqueeze(1).repeat((B, T)).reshape((B, D, T))
            norm_spec = (spectrogram - spectrogram_mean) + global_mean
        else:
            norm_spec = spectrogram - spectrogram_mean
            
        return norm_spec


class _AxisMasking(torch.nn.Module):
    r"""Apply masking to a spectrogram.

    Args:
        mask_param (int): Maximum possible length of the mask.
        axis (int): What dimension the mask is applied on.
        iid_masks (bool): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 4D.
    """
    __constants__ = ['mask_param', 'axis', 'iid_masks']

    def __init__(self, mask_param: int, axis: int, iid_masks: bool, prob: float) -> None:

        super(_AxisMasking, self).__init__()
        assert 0.0 < prob <= 1.0, "prob must be less than 1.0 and greater than 0.0 !"
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks
        self.prob = prob

    def forward(self, specgram: Tensor, mask_value: float = -0.0) -> Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of dimension (..., freq, time).
            mask_value (float): Value to assign to the masked columns.

        Returns:
            Tensor: Masked spectrogram of dimensions (..., freq, time).
        """
        # if iid_masks flag marked and specgram has a batch dimension
        if np.random.uniform(0,1) < self.prob:
            if self.iid_masks and specgram.dim() == 4:
                return F.mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
            else:
                return F.mask_along_axis(specgram, self.mask_param, mask_value, self.axis)
        else:
            return specgram


class FrequencyMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the frequency domain.

    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, freq_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
    """
    def __init__(self, freq_mask_param: int, iid_masks: bool = False, prob: float = 0.3) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks, prob)


class TimeMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the time domain.

    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
    """
    def __init__(self, time_mask_param: int, iid_masks: bool = False, prob: float = 0.3) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks, prob)


class ReShape(torch.nn.Module):
    """
        Convert tensor of shape (batch, freq, time) -> (batch, time, freq)
    """

    def __init__(self) -> None:
        super(ReShape, self).__init__()


    def forward(self, spectrogram: Tensor) -> Tensor:
        r"""
        Args:
            spectrogram (Tensor): Tensor shape of `(..., freq, time)`.

        Returns:
            Tensor: Tensor shape of `(..., time, freq)`.
        """

        return spectrogram.permute(dims=(0,2,1))
