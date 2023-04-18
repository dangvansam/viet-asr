import time
import math
from typing import Iterator

import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import pad

from preprocessing.pre_utils import *
from preprocessing.augmentation import Augmentor


def mask_label(
    unk_id: int,
    input_decoder_id: list,
    prob: float = 0.1,
) -> list:

    N = len(input_decoder_id)
    num_mask = round(0.2*N + 0.5)

    if N > 4:

        if np.random.uniform(0, 1) < prob:
            tmp = list(range(1, N))     # except start token
            random.shuffle(tmp)

            # random replace with unk token
            for i in tmp[0:num_mask]:
                input_decoder_id[i] = unk_id

    return input_decoder_id


class ASR_Dataset(Dataset):

    def __init__(self,
                 meta_dictionary: dict,
                 bpe_model_path: str,
                 type_dataset: str,
                 train_dev_ratio: float = 0.95,
                 hop_length: int = 160,
                 sample_rate: int = 16000,
                 noise_folder_path: str = None,
                 augt_prob: float = 0.7,
                 on_fly_augt: bool = True,
                 *args, **kwargs
                 ) -> None:

        self.train = True
        self.meta_dictionary = meta_dictionary

        self.wav_paths = list(self.meta_dictionary)
        total_paths = len(self.wav_paths)

        random.seed(6868)
        random.shuffle(self.wav_paths)

        break_point = int(train_dev_ratio * total_paths)

        if type_dataset == "train":
            self.wav_paths = self.wav_paths[0:break_point]
            pass
        elif type_dataset == "valid":
            self.wav_paths = self.wav_paths[break_point:total_paths]
            self.train = False
        else:
            raise ValueError(
                "type_dataset does not exist, please choose 'train', 'train_pseudo' or 'valid'!")

        # load bpe model
        assert os.path.isfile(bpe_model_path), "BPE model is not exists!"
        self.bpe_model = spm.SentencePieceProcessor(model_file=bpe_model_path)

        self.encoder_vocab = self.bpe_model.GetPieceSize()
        self.sil_index = self.bpe_model.pad_id()
        self.pad_id = self.bpe_model.eos_id()
        self.unk_id = self.bpe_model.unk_id()
        self.end_of_word = self.bpe_model.bos_id()

        # augmentation
        self.augt_prob = augt_prob
        self.on_fly_augt = on_fly_augt
        self.augt = Augmentor(noise_folder_path)
        print("on_fly_augt: ", self.on_fly_augt)

        self.length = len(self.wav_paths)
        self.data = list(range(self.length))
        self.data = np.array(self.data)

    def assign_label(
        self,
        input_decoder_id: str,
        output_decoder_id: str,
        duration: float
    ) -> Tuple[None, list, list]:
        return None, input_decoder_id, output_decoder_id

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        wave_path_index = self.data[index]
        wav_path = self.wav_paths[wave_path_index]
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        # gen label
        wav_infor_dict = self.meta_dictionary[wav_path]
        label, input_decoder_id, output_decoder_id = self.assign_label(**wav_infor_dict)

        if self.train and self.on_fly_augt:
            waveform = self.augt.augt(waveform, prob=self.augt_prob)

        input_decoder_id = torch.tensor(input_decoder_id)
        output_decoder_id = torch.tensor(output_decoder_id)

        return waveform, label, self.pad_id, input_decoder_id, output_decoder_id


class ASR_Dataset_Alignments(Dataset):

    def __init__(self,
                 word_alignments: dict,
                 bpe_model_path: str,
                 type_dataset: str,
                 train_dev_ratio: float = 0.95,
                 hop_length: int = 160,
                 sample_rate: int = 16000,
                 *args, **kwargs
                 ) -> None:

        # load word alignments
        self.hop_length = hop_length
        self.hop_length_seconds = float(hop_length/sample_rate)
        self.min_wave_length = 2 * int(sample_rate)

        self.train = True
        self.word_alignments = word_alignments

        self.wav_paths = list(self.word_alignments)
        total_paths = len(self.wav_paths)

        random.seed(6868)
        random.shuffle(self.wav_paths)

        break_point = int(train_dev_ratio * total_paths)

        if type_dataset == "train":
            self.wav_paths = self.wav_paths[0:break_point]
        elif type_dataset == "valid":
            self.wav_paths = self.wav_paths[break_point:total_paths]
            self.train = False
        else:
            raise ValueError(
                "type_dataset does not exist, please choose 'train' or 'valid'!")

        # load bpe model
        assert os.path.isfile(bpe_model_path), "BPE model is not exists!"
        self.bpe_model = spm.SentencePieceProcessor(model_file=bpe_model_path)

        self.encoder_vocab = self.bpe_model.GetPieceSize()
        self.sil_index = self.bpe_model.pad_id()
        self.pad_id = self.bpe_model.eos_id()
        self.unk_id = self.bpe_model.unk_id()
        self.end_of_word = self.bpe_model.bos_id()

        # self.wav_paths = sorted(self.wav_paths, key= lambda x: word_alignments[x][-1]["end"])
        self.length = len(self.wav_paths)
        self.data = list(range(self.length))
        self.data = np.array(self.data)

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def parse_word_infor(self,
                         word: str,
                         start: float,
                         end: float
                         ) -> tuple:
        return word, start, end

class ASR_Test_Dataset(Dataset):

    def __init__(self,
                 bpe_model_path: str,
                 meta_test_path: str
                 ) -> None:

        # load bpe model
        assert os.path.isfile(bpe_model_path), "BPE model is not exists!"
        self.bpe_model = spm.SentencePieceProcessor(model_file=bpe_model_path)

        self.vocab = self.bpe_model.GetPieceSize()
        self.sil_index = self.bpe_model.pad_id()
        self.pad_id = self.bpe_model.eos_id()

        self.bos_id = self.bpe_model.bos_id()
        self.eos_id = self.bpe_model.eos_id()
        self.end_of_word = self.bpe_model.bos_id()

        assert os.path.isfile(meta_test_path), "meta_test_path is not exists!"
        self.meta = self.load_meta(meta_test_path)

        self.length = len(self.meta)
        self.data = list(range(self.length))
        self.data = np.array(self.data)

    def load_meta(self, meta_test_path: str) -> list:

        data = list()

        with open(meta_test_path, encoding='utf8') as fp:
            for line in fp:
                tokens = line.strip().split("|")

                if len(tokens) != 3:
                    print('missing line: {line}'.format(line=line))
                    continue

                wav_path, transcript, dur = tokens
                dur = ast.literal_eval(dur)

                if dur < 1:
                    print("{wav_path} is too short!".format(wav_path=wav_path))
                    continue

                if '111' in transcript:
                    print("{wav_path}, 111 in script!".format(
                        wav_path=wav_path))
                    continue

                transcript = self.normalize_label(transcript)
                new_line = "{path}|{label}|{dur}".format(
                    path=wav_path, label=transcript, dur=dur)

                data.append(new_line)

        return data

    def normalize_label(self, label: str) -> str:
        label = label.upper()
        label = label.replace(" CK", "").replace("SPKT", "")
        label = re.sub("[^\ws]", " ", label)
        label = re.sub("\s+", " ", label)
        return label.strip()

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):

        meta_index = self.data[index]
        meta_line = self.meta[meta_index]
        wav_path, transcript, dur = meta_line.split("|")

        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        return waveform, wav_path, transcript, dur


class Mix_Dataset(Dataset):

    """
        Mix labeled dataset and pseudo-label dataset loader
    """

    def __init__(
        self,
        dt1: DataLoader,
        dt2: DataLoader,
    ) -> None:

        super(Mix_Dataset, self).__init__()

        self.dt1_length = len(dt1)
        self.dt2_length = len(dt2)

        self.dt1, self.dt2 = dt1, dt2

        total_length = self.dt1_length + self.dt2_length

        self.data = list(range(total_length))
        self.data = np.array(self.data)

        self.dataloaders1, self.dataloaders2 = iter(dt1), iter(dt2)

    def get_batch(
        self,
        iterator: Iterator,
        use_dt1: bool,
        use_dt2: bool,
    ):

        try:
            data = next(iterator)
        except Exception as _:
            self.dataloaders1, self.dataloaders2 = iter(
                self.dt1), iter(self.dt2)

            if use_dt1:
                iterator = self.dataloaders1
            elif use_dt2:
                iterator = self.dataloaders2

            data = next(iterator)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if index < self.dt1_length:
            batch = self.get_batch(
                self.dataloaders1, use_dt1=True, use_dt2=False)
            pseudo_label = False
        else:
            batch = self.get_batch(
                self.dataloaders2, use_dt1=False, use_dt2=True)
            pseudo_label = True

        if batch is None:
            raise ValueError("Batch is None! Check your iterator!")

        return batch, pseudo_label


class ASR_Dataset_SSL(Dataset):

    def __init__(
        self,
        wav_paths: list,
        type_dataset: str,
        train_dev_ratio: float = 0.95
    ) -> None:

        self.wav_paths = wav_paths
        total_paths = len(self.wav_paths)
        self.train = False

        random.seed(6868)
        random.shuffle(self.wav_paths)

        break_point = int(train_dev_ratio * total_paths)

        if type_dataset == "train":
            self.wav_paths = self.wav_paths[0:break_point]
            self.train = True

        elif type_dataset == "valid":
            self.wav_paths = self.wav_paths[break_point:total_paths]

        else:
            raise ValueError(
                "type_dataset does not exist, please choose 'train' or 'valid'!")

        self.wav_paths = sorted(self.wav_paths, key=lambda x: x[1])
        self.wav_paths = [x[0] for x in self.wav_paths]

        self.length = len(self.wav_paths)
        self.data = list(range(self.length))
        self.data = np.array(self.data)

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):

        wave_path_index = self.data[index]

        if self.train:
            min_range = max(0, wave_path_index - 64)
            max_range = min(self.length, wave_path_index + 64)
            wave_path_index = np.random.randint(min_range, max_range)

        wav_path = self.wav_paths[wave_path_index]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        return waveform
