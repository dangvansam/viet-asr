import os, sys
sys.path.append(os.getcwd())

import torch
import torchaudio
from torch import Tensor

from .transform import ReShape, MelSpectrogram, FrequencyMasking, TimeMasking, AmplitudeToDB, RangeNorm, CMVN

from utils import load_json, pad_waveform



class Extractor(torch.nn.Module):
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
            n_fft: int = 1024,
            freq_mask_prob: float = 0.8,
            time_mask_prob: float = 0.8,
            global_cmvn_path: str = None,
            device: str = 'cpu',
            use_specaugt: bool = True,
            use_global_norm: bool = False,
            *args, **kwargs
        ):

        super(Extractor, self).__init__()

        self.hop_length = int( (window_size-stride) * sample_rate)
        self.num_mels = num_mels
        self.use_specaugt = use_specaugt

        # load global mean value
        global_mean_infor = None
        
        if use_global_norm:
            global_mean_infor = load_json(global_cmvn_path)

        extractor = [
                MelSpectrogram(sample_rate = sample_rate, n_fft= n_fft, hop_length= self.hop_length, n_mels= self.num_mels),
                AmplitudeToDB(),
                RangeNorm()
            ]
        
        specAugment = [
                FrequencyMasking(freq_mask_param, prob= freq_mask_prob),
                FrequencyMasking(freq_mask_param, prob= freq_mask_prob),
                TimeMasking(time_mask_param, prob= time_mask_prob),
                TimeMasking(time_mask_param, prob= time_mask_prob)
            ]

        self.extractor = torch.nn.Sequential(*extractor)
        self.specAugment = torch.nn.Sequential(*specAugment)

        self.norm = CMVN(global_mean_infor, device, global_norm= use_global_norm)
        self.reshape = ReShape()

    def forward(
            self, 
            waveform: Tensor,
        ) -> Tensor:
        
        ''' Features Extraction foward
        
        Args:
            - waveform (Tensor): Batch of waveform inputs (batch, channel, time)

        Outputs:
            - Mel-spectrogram (Tensor): Batch of mel-spectrogram (batch, num_frames, num_mels)
        '''

        with torch.no_grad():
            B, T = waveform.shape
            max_frames = int(T/self.hop_length)

            features = self.extractor(waveform)
            
            if self.training and self.use_specaugt:
                features = self.specAugment(features)
            
            features = self.norm(features)
            features = self.reshape(features)
            features = features[:, :max_frames, :]

        return features



def test(wav_path: str = "test.wav"):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    waveform, sr = torchaudio.load(wav_path)
    waveform = pad_waveform([waveform], min_audio_seconds= 5)
    waveform = waveform[0].unsqueeze(dim=0)

    extractor = Extractor(global_cmvn_path= "dataset/global_cmvn_infor.json", use_specaugt= False)

    print(waveform.shape)
    features = extractor(waveform)
    print(features.shape, features)

    feat = features[0]
    array_feat = feat.numpy()

    plt.figure()
    plt.imshow(array_feat, cmap='Greens')
    plt.gca().invert_yaxis()
    plt.show()

    return features