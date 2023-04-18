import os, glob
from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import pad

import kenlm
import sentencepiece as spm

from preprocessing.data import ASR_Test_Dataset
from preprocessing.pre_utils import Test_Loading
from preprocessing import data

from models import asr_model, transducer, ensemble_ssl
from models.wavlm.model import WavLM, WavLMConfig

from config import load_config


def my_collate(batch) -> Tuple[Tensor, list]:

    waveform = list()
    audio_paths = list()

    for wform, wav_path in batch:
        waveform.append(wform)
        audio_paths.append(wav_path)

    waveform = pad_waveform(waveform)
    waveform = torch.stack(waveform, dim= 0)

    return [waveform, audio_paths]

def my_collate_v2(batch) -> Tuple[Tensor, list, list, list]:

    waveform = list()
    audio_paths = list()
    label = list()
    duration = list()

    for wform, wav_path, transcript, dur in batch:
        waveform.append(wform)
        audio_paths.append(wav_path)
        label.append(transcript)
        duration.append(dur)

    waveform = pad_waveform(waveform)
    waveform = torch.stack(waveform, dim= 0)

    return [waveform, audio_paths, label, duration]
    

def build_convert_dict(
            lexicon_path: str,
        ) -> dict:

    data = dict()

    if not lexicon_path:
        return None

    with open(lexicon_path, encoding='utf8') as fp:
        for line in fp:
            line = line.strip()
            tokens = line.split()

            word = tokens[0]
            word_units = tokens[1:]
            units_seq = ''.join(word_units)

            if units_seq not in data:
                data[units_seq] = (word,)
            else:
                data[units_seq] += (word,)
    
    return data



def pad_waveform(
        sequence_waveform: List[torch.Tensor],
        min_audio_seconds: int = 2,
        sample_rate: int = 16000,
        padding_value: float = 0.0
    ) -> List[torch.Tensor]:

    '''Padding list of 2D Tensor
    '''

    output = list()

    min_length = min_audio_seconds * sample_rate
    wav_lengths = [x.shape[1] for x in sequence_waveform]
    max_length = max(min_length, max(wav_lengths))

    for tensor in sequence_waveform:
        tensor_length = tensor.shape[1]
        pad_length = max_length - tensor_length
        tensor = pad(tensor, (0, pad_length), value= padding_value)
        output.append(tensor)
    
    return output


def convert_2_char(
        index_seq: list,
        chars: list
    ) -> list:

    data = list()
    length = len(chars)

    for x in index_seq:
        if x >= length or x < 0:
            break
        data.append(chars[x])

    return data



class BpeProcessor(torch.nn.Module):

    def __init__(
            self, 
            bpe_model_path: str
        ) -> None:

        super(BpeProcessor, self).__init__()
        self.bpe_model = spm.SentencePieceProcessor(model_file= bpe_model_path)

    def extract_keywords(
                self, 
                x: Union[List[str], List[int], str]
            ) -> str:

        if type(x) == str:
            x = x.split()

        output = self.bpe_model.Decode(x)

        return output


def build_search_processor(
            bpe_model_path: str = None
    ) -> BpeProcessor:

    kp = BpeProcessor(bpe_model_path)
    return kp


def kenlm_rescore(
        kenlm_model: kenlm.Model,
        w_founded: Union[List[tuple], str],
        acoustic_score: int,
        lm_weight: float = 0.7
    ) -> List[tuple]:

    acoustic_weight = 1 - lm_weight

    candidates = [(0,"")]

    if len(w_founded) == 0:
        candidates = [(0,"")]

    elif len(w_founded) > 1:

        if type(w_founded) == list:
            sentence = product_candidates(w_founded, kenlm_model)
        elif type(w_founded) == str:
            sentence = w_founded

        length = len(sentence.split())
        sentence_score = lm_weight * kenlm_model.score(sentence, bos = True, eos = False) + acoustic_weight * acoustic_score
        sentence_score = sentence_score / length
        candidates = [(sentence_score, sentence)]        
    
    return candidates


def product_candidates(
        w_founded: List[tuple],
        kenlm_model: kenlm.Model
    ) -> str:

    sentence = ""

    for words in w_founded:
        temp = list()

        if len(words) == 1:
            sentence = "{} {}".format(sentence, words[0])
        else:
            for w in words:
                new_sentence = sentence + " " + w
                score = kenlm_model.score(new_sentence, bos = True, eos = False)
                temp.append((score, new_sentence))

            best_sentence = max(temp, key= lambda x: x[0])
            sentence = best_sentence[1]

    return sentence.strip()


def average_snapshots(
        list_of_snapshots_paths: list,
        device: str,
        type_model: str,
        vocab: int,
        model_params: dict
    ) -> torch.nn.Module:

    snapshots_weights = {}

    for snapshot_path in list_of_snapshots_paths:
        
        model = model.ASR(vocab, model_params)

        model.load_state_dict(torch.load(snapshot_path, map_location= torch.device(device)), strict= False)
        snapshots_weights[snapshot_path] = model

    avg_state_dict = None
    N = len(snapshots_weights)

    for snapshot_path in snapshots_weights:
        if avg_state_dict is None:
            avg_state_dict = snapshots_weights[snapshot_path].state_dict()
        else:
            current_state_dict = snapshots_weights[snapshot_path].state_dict()

            for key in avg_state_dict:
                if type(avg_state_dict[key]) == dict: continue
                avg_state_dict[key] = avg_state_dict[key] + current_state_dict[key]
    
    for key in avg_state_dict:
        if type(avg_state_dict[key]) != dict:
            if str(avg_state_dict[key].dtype).startswith("torch.int"):
                pass
            else:
                avg_state_dict[key] = avg_state_dict[key] / N
    
    avg_model = model.ASR(vocab, model_params)

    avg_model.load_state_dict(avg_state_dict)

    return avg_model

def update_results(
            test_transcript: dict,
            batch_transcript: list,
            audio_paths: list,
        ) -> None:
    
    print("\n")

    for i, text in enumerate(batch_transcript):
        path = audio_paths[i]
        test_transcript[path] = text
        print("{}: {}".format(path, text))


def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def gen_mask(sz: int) -> Tensor:
    x = torch.zeros(sz, sz)
    
    for i in range(sz):
        if i == sz: continue
        x[i][i+1:] = float('-inf')
    return x


def build_testset_loader(
        bpe_model_path: str,
        meta_test_path: dict,
    ) -> Tuple[data.ASR_Test_Dataset, int]:

    testset = data.ASR_Test_Dataset(bpe_model_path, meta_test_path)
    vocab = testset.vocab

    return testset, vocab

def init_ensemble_ssl(
        vocab: int,
        model_params: dict,
        device: str
    ):
    config_paths = model_params['config_paths']
    model_state_dict_path = model_params['model_state_dict_path']
    ssl_models = torch.nn.ModuleList()

    for config, state_dict_path in zip(config_paths, model_state_dict_path):
        tmp_config = load_config.load_yaml(config)

        model = model.ASR(vocab, tmp_config['model'])
        # model.load_state_dict(torch.load(state_dict_path, map_location= torch.device(device)))
        ssl_models.append(model)

    model = ensemble_ssl.Encoder(vocab, model_params['d_models'], ssl_models)
    return model

def load_model(
        checkpoint_folder: str,
        vocab: int,
        model_params: dict,
        device: str,
        model_state_dict_path: str,
        type_model: str = "seq2seq", # transducer or seq2seq
        num_avg: int = 5,
        *args, **kwargs
    ) -> Union[asr_model.ASR, transducer.ASR]:

    if not checkpoint_folder and not model_state_dict_path:
        raise ValueError('checkpoint_folder or model_state_dict_path is require!')

    snapshots_paths = glob.glob(os.path.join(checkpoint_folder, "*.pt"))
    snapshots_paths = [path for path in snapshots_paths if "infer_model" not in path]
    snapshots_times = [os.path.getmtime(p) for p in snapshots_paths]
    snapshots_paths = [p for time, p in sorted(zip(snapshots_times, snapshots_paths), key= lambda x: x[0], reverse= True)]

    if checkpoint_folder:
        print("\n Assemble checkpoints: {} \n".format(snapshots_paths[0:num_avg]))
        model = average_snapshots(snapshots_paths[0:num_avg], device, type_model, vocab, model_params)
        infer_model_path = os.path.join(checkpoint_folder, "infer_model_state_dict.pt")
        torch.save(model.state_dict(), infer_model_path)
        full_model_path = os.path.join(checkpoint_folder, "infer_model.pt")
        torch.save(model, full_model_path)
        print(f"\nSaved {full_model_path}")
    else:
        model = model.ASR(vocab, model_params)
        model.load_state_dict(torch.load(model_state_dict_path, map_location= torch.device(device)))

    model = model.to(device)
    model.eval()

    return model