import torch
from torch import Tensor

from utils import load_json
from vietasr.model.model_utils import make_pad_mask

from vietasr.model.layers.transform import (
    CMVN,
    AmplitudeToDB,
    FrequencyMasking,
    MelSpectrogram,
    RangeNorm,
    TimeMasking
)

class FeatureExtractor(torch.nn.Module):
    r"""Extract Mel-Spectrogram from the raw waveform and normalized
    """

    def __init__(
        self,
        num_mels: int = 80,
        stride: float = 0.015,
        window_size: float = 0.025,
        freq_mask_param: int = 20,
        time_mask_param: int = 20,
        sample_rate: int = 16000,
        n_fft: int = 512,
        freq_mask_prob: float = 0.8,
        time_mask_prob: float = 0.8,
        global_cmvn_path: str = None,
        use_specaugt: bool = True,
        use_global_norm: bool = False,
    ):

        super(FeatureExtractor, self).__init__()

        self.hop_length = int((window_size-stride) * sample_rate)
        self.num_mels = num_mels
        self.n_fft = n_fft
        self.use_specaugt = use_specaugt

        # load global mean value
        global_mean_infor = None

        if use_global_norm:
            global_mean_infor = load_json(global_cmvn_path)

        extractor = [
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=self.hop_length,
                n_mels=self.num_mels
            ),
            AmplitudeToDB(),
            RangeNorm()
        ]

        specAugment = [
            FrequencyMasking(freq_mask_param, prob=freq_mask_prob),
            FrequencyMasking(freq_mask_param, prob=freq_mask_prob),
            TimeMasking(time_mask_param, prob=time_mask_prob),
            TimeMasking(time_mask_param, prob=time_mask_prob)
        ]

        self.extractor = torch.nn.Sequential(*extractor)
        self.specAugment = torch.nn.Sequential(*specAugment)

        self.norm = CMVN(global_mean_infor, global_norm=use_global_norm)

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor = None
    ) -> Tensor:
        ''' Features Extraction foward

        Args:
            - inputs (Tensor): Batch of waveform inputs (batch, time)
            - input_lengths (Tensor): Batch of waveform input lengths (batch)

        Outputs:
            - Mel-spectrogram (Tensor): Batch of mel-spectrogram (batch, num_frames, num_mels)
        '''

        with torch.no_grad():
            feats = self.extractor(inputs)
            if self.training and self.use_specaugt:
                feats = self.specAugment(feats)
            feats = feats.transpose(1, 2)
            if input_lengths is not None:
                pad = self.n_fft // 2
                input_lengths = input_lengths + 2 * pad
                feat_lens = (input_lengths - self.n_fft) // self.hop_length + 1
                feats.masked_fill_(make_pad_mask(feat_lens, feats, 1), 0.0)
            else:
                feat_lens = None
            feats = self.norm(feats)

        return feats, feat_lens