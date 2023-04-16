import os, time, glob, math

import random
from typing import Callable, List
import numpy as np

import torch
from torch import Tensor
import torchaudio



def apply_effects(func):

    def wrap(*args, **kwargs):
        effects, waveform, sample_rate = func(*args, **kwargs)
        # print(func.__name__)
        new_waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects, channels_first=True)
        return new_waveform, sample_rate

    return wrap


class Augmentor:

    def __init__(
            self,
            noise_folder_path: str,
            rir_folder_path: str,
        ) -> None:

        super(Augmentor, self).__init__()

        self.noise_paths = glob.glob(os.path.join(noise_folder_path, "*.wav"))
        self.rir_paths = glob.glob(os.path.join(rir_folder_path, "*.wav"))

        self.num_noise_paths = len(self.noise_paths)
        self.num_rir_paths = len(self.rir_paths)
        self.augt_funcs = [self.rir_wav, self.change_pitch, self.change_tempo, self.add_noise]


    def random_from_range(
            self, 
            min_value: float, 
            max_value: float,
        ):

        return str(round(np.random.uniform(min_value, max_value),2))


    @apply_effects
    def change_pitch(
            self,
            waveform: Tensor,
            sample_rate: int = 16000,
            min_pitch: int = 150,
            max_pitch: int = 300,
        ) -> list:

        effects = list()
        pitch_factor = self.random_from_range(min_pitch, max_pitch)

        if np.random.uniform(0,1) < 0.5:
            pitch_factor = str(- float(pitch_factor))
        
        effects.append(["pitch", pitch_factor])
        effects.append(['rate', str(sample_rate)])
        return effects, waveform, sample_rate


    @apply_effects
    def change_tempo(
            self,
            waveform: Tensor,
            sample_rate: int = 16000,
            min_tempo: int = 0.9,
            max_tempo: int = 1.1,
        ) -> list:

        effects = list()
        tempo_factor = self.random_from_range(min_tempo, max_tempo)

        effects.append(["tempo", tempo_factor])
        effects.append(['rate', str(sample_rate)])
        return effects, waveform, sample_rate


    def add_noise(
            self,
            speech: Tensor,
            samplerate: int = 16000,
            min_snr_db: int = 3,
            max_snr_db: int = 15,
        ):

        # begin = time.time()
        index = np.random.randint(low= 0, high= self.num_noise_paths, size= (1,))[0]
        
        noise, sr = torchaudio.load(self.noise_paths[index])
        
        speech_length = speech.shape[1]
        noise_length = noise.shape[1]

        if noise_length >= speech_length:
            noise = noise[:, :speech_length]
        else:
            num_repeat = round(speech_length/noise_length + 0.5)
            noise = noise.repeat(1, num_repeat)
            noise = noise[:, :speech_length]

        # print("\ntime load noise:", time.time() - begin)

        # begin = time.time()
        speech_power = speech.norm(p=2)
        noise_power = noise.norm(p=2)
        # print("time norm:", time.time() - begin)
        
        # begin = time.time()

        snr_db = np.random.randint(low= min_snr_db, high= max_snr_db, size= (1,))[0]
        snr = math.exp(snr_db/10)
        scale = snr * noise_power / speech_power
        noisy_speech = (scale * speech + noise) / 2
        # print("time add noise:", time.time() - begin)

        return noisy_speech, samplerate


    def rir_wav(
            self,
            waveform: Tensor,
            sample_rate: int = 16000,   
        ):

        index = np.random.randint(low= 0, high= self.num_rir_paths, size= (1,))[0]
        rir_raw, sr = torchaudio.load(self.rir_paths[index])

        rir_raw = rir_raw[:, int(sample_rate*0):int(sample_rate*0.1)]
        rir = rir_raw / torch.norm(rir_raw, p=2)
        rir = torch.flip(rir, [1])

        # begin = time.time()
        speech_ = torch.nn.functional.pad(waveform, (rir.shape[1]-1, 0))
        speech_, rir = speech_.cuda(), rir.cuda()
        augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
        speech_, rir = speech_.detach().cpu(), rir.detach().cpu()
        augmented = augmented.cpu()
        # print("time convolution:", time.time() - begin)

        return augmented, sample_rate


    def augt(
            self,
            waveform: Tensor,
            samplerate: int = 16000,
            prob: float = 0.8,
        ) -> Tensor:

        assert 0 < prob < 1.0, "prob must be in range (0,1) !"

        try:
            if np.random.uniform(0,1) < prob:
                apply_func = random.choice(self.augt_funcs)
                new_waveform, sr = apply_func(waveform, samplerate)
            else:
                new_waveform = waveform
        except Exception as ex:
            new_waveform = waveform
        
        return new_waveform


def test():
    augmentor = Augmentor(noise_folder_path="dataset/audio/noise/general", rir_folder_path= "dataset/audio/noise/rir/wavs")

    bgin = time.time()

    for i in range(1000):
        waveform, sr = torchaudio.load("test.wav")
        new_waveform = augmentor.augt(waveform, sr)
        # torchaudio.save("test_augt.wav", new_waveform, sr)

    print("time: ", time.time() - bgin)