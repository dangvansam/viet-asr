import os, glob, ast, re, random
from typing import Tuple

import torch
import audiofile as af
import numpy as np

import sentencepiece as spm
from torch.nn.functional import pad

def pad_waveform_v2(
        waveform: torch.Tensor,
        s1: int = 2,
        s2: int = 1,
        sd: int = 2,
        padding_value: float = 0.0,
        hop_length: int = 160
    ):

    a = 3 + 2*s1 + 2*s1*s2 + 2*s1*s2*sd
    b = s1*s2*sd*sd
    C, T = waveform.shape

    k = int(T/(b*hop_length)) + 1
    new_T = (k*b+a) * hop_length
    pad_length = new_T - T

    # print(a, b, new_T, pad_length)
    waveform = pad(waveform, (0, pad_length), value= padding_value)
    return waveform

class Base_Loading:

    def __init__(
            self,
            bpe_model_path: str
        ) -> None:

        super(Base_Loading, self).__init__()

        assert os.path.isfile(bpe_model_path), "BPE model is not exists!"
        self.bpe_model = spm.SentencePieceProcessor(model_file= bpe_model_path)


    def load_bpe_char(
            self,
            bpe_vocab_path: str
        ) -> list:

        chars = list()

        with open(bpe_vocab_path, encoding='utf8') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    bpe_unit = line.split("\t")[0]
                    chars.append(bpe_unit)
        
        return chars


class Test_Loading(Base_Loading):

    def __init__(self, bpe_model_path: str) -> None:
        super(Test_Loading, self).__init__(bpe_model_path)
    

    def load_wav_path_infer(
            self,
            path: str
        ) -> list:

        assert os.path.isfile(path), "Input test scp file is not exists!"
        meta = list()

        with open(path, encoding="utf8") as fp:
            for line in fp:
                line = line.strip()

                if not line: continue

                utt_id, wav_path = line.strip().split()

                if os.path.isfile(wav_path):
                    if af.duration(wav_path) < 1: continue
                    meta.append(wav_path)
                else:
                    print("File with path {} is not exits!".format(wav_path))
        return meta


class Train_Loading(Base_Loading):

    def __init__(
            self,
            meta_folder: str,
            bpe_model_path: str,
        ) -> None:

        super(Train_Loading, self).__init__(bpe_model_path)

        assert os.path.isdir(meta_folder), "Meta folder is not exists!"
        self.meta_paths = glob.glob(os.path.join(meta_folder, "*.txt"))

    
    def normalize_label(self, label: str) -> str:
        label = label.upper()
        label = label.replace(" CK", "").replace("SPKT", "")
        label = re.sub("[^\ws]", " ", label)
        label = re.sub("\s+", " ", label)
        return label.strip()
    

    def label_regularization(
            self,
            input_decoder_id: list,
            unk_index: int, 
            prob: float = 0.15,
        ) -> list:

        N = len(input_decoder_id)
        max_num_mask = round(0.3*N + 0.5)

        if N > 2:
        
            if np.random.uniform(0,1) < prob:
                # random swap two label index
                i, j = np.random.randint(low=1, high=N, size=(2,))
                input_decoder_id[i], input_decoder_id[j] = input_decoder_id[j], input_decoder_id[i]
            
            if np.random.uniform(0,1) < prob:
                tmp = list(range(N))
                random.shuffle(tmp)
                num_mask = np.random.randint(low=0, high= max_num_mask, size=(1,))[0]

                # random replace with unk token
                for i in tmp[0:num_mask]:
                    input_decoder_id[i] = unk_index
        
        return input_decoder_id


    def make_bpe_label(
            self,
            bpe_model: spm.SentencePieceProcessor,
            label: str
        ) -> Tuple[list, list, bool]:

        oov = False

        if len(label.split()) == 1:
            oov = True

        input_decoder_id = bpe_model.Encode(label, out_type= int, add_bos= True, add_eos= False)
        output_decoder_id = bpe_model.Encode(label, out_type= int, add_bos= False, add_eos= True)
        # sp.GetPieceSize()

        return input_decoder_id, output_decoder_id, oov
    

    def filter_audio(
            self,
            wav_path: str,
            input_decoder_id: list,
            duration: float,
            label: str,
            min_dur: float = 0.8,
            max_dur: float = 15
        ) -> bool:
        
        filtered = False

        con1 = "111" in label
        con2 = 25 * duration < len(input_decoder_id) - 1
        con3 = duration > max_dur or duration < min_dur
        con4 = not os.path.isfile(wav_path)
        # con5 = self.unk_index in input_decoder_id

        if con1 or con2 or con3 or con4:
            filtered = True
        
        return filtered


    def train_dev_split(
            self,
            data_training: dict,
            training_data_ratio: float = 0.95
        ) -> Tuple[list, list]: 

        data_train = list()
        data_dev = list()
        
        for key, paths in data_training.items():
            if not paths: continue

            random.seed(6868)
            random.shuffle(paths)
            
            length = len(paths)
            break_point = int(length * training_data_ratio)
            train = paths[:break_point]
            dev = paths[break_point:]

            data_train.extend(train)
            data_dev.extend(dev)
            print("\n{}: \n _Number of audio train: {}, \n _Number of audio dev: {}".format(key, len(train), len(dev)))
        
        return data_train, data_dev

    def load_meta(self) -> dict:

        meta_dictionary = dict()

        for path in self.meta_paths:
            with open(path, encoding='utf8') as fp:
                for line in fp:
                    line = line.strip()
                    if not line: continue

                    wav_path, label, duration = line.split('|')
                    duration = ast.literal_eval(duration)
                    label = self.normalize_label(label)

                    input_decoder_id, output_decoder_id, oov = self.make_bpe_label(self.bpe_model, label= label)
                    filtered = self.filter_audio(wav_path, input_decoder_id, duration, label)
                    if filtered or oov: continue

                    meta_dictionary[wav_path] = {
                        'input_decoder_id': input_decoder_id, 
                        'output_decoder_id': output_decoder_id, 
                        'duration': duration
                    }

        return meta_dictionary
