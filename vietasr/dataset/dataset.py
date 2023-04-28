
from typing import List, Tuple, Union

import os
import torch
import torchaudio
from loguru import logger
from torch.utils.data import Dataset
from vietasr.dataset.tokenizer import SentencepiecesTokenizer
from utils import pad_list

class ASRDataset(Dataset):
    def __init__(self, meta_filepath: Union[str, List[str]]):
        if isinstance(meta_filepath, str):
            meta_filepath = [meta_filepath]
        data = []
        for filepath in meta_filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().split("|")
                    wav_filepath = line[0]
                    text = line[1]
                    dur = float(line[2])
                    if dur > 12:
                        logger.warning(f"skipped long file, duration={dur}")
                        continue
                    if not os.path.exists(wav_filepath):
                        logger.error(f"file is not exists: {wav_filepath}")
                        exit()
                    if len(text.strip()) == 0:
                        continue
                    data.append((line[0], line[1]))
        logger.info(f"loaded {len(data)} samples")
        print(data[:0])
        
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
class ASRCollator():
    def __init__(
        self,
        bpe_model_path: str
    ):
        self.tokenizer = SentencepiecesTokenizer(bpe_model_path)
        vocab = self.tokenizer.get_vocab()
        vocab = vocab[3:]
        vocab = ["<blank>", "<unk>"] + vocab + ["<pad>"]
        self.vocab = vocab
        self.token2ids = {t:i for i,t in enumerate((vocab))}
        self.ids2token = {i:t for i,t in enumerate((vocab))}
        self.blank_id = 0
        self.unk_id = 1
        self.pad_id = len(vocab) - 1
    
    def get_vocab(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def text2ids(self, text: str):
        tokens = self.tokenizer.text2tokens(text)
        ids = [self.token2ids.get(t, self.unk_id) for t in tokens]
        return ids
        
    def ids2text(self, ids: List[int]):
        tokens = [self.ids2token[i] for i in ids if i not in [self.blank_id, self.unk_id, self.pad_id]]
        text = self.tokenizer.tokens2text(tokens)
        return text
        
    def __call__(self, batch: Tuple[str, str]):
        inputs = []
        input_lens = []
        targets = []
        target_lens = []
        for b in batch:
            waveform = torchaudio.load(b[0])[0]
            waveform = waveform.squeeze(0)
            if waveform.shape[0] == 2:
                waveform = waveform[0]
            inputs.append(waveform)
            input_lens.append(waveform.shape[0])

            target = torch.LongTensor(self.text2ids(b[1]))
            targets.append(target)
            target_lens.append(target.shape[0])
        
        
        inputs = pad_list(inputs, pad_value=0.0)
        targets = pad_list(targets, pad_value=self.pad_id)
        input_lens = torch.LongTensor(input_lens)
        
        return inputs, input_lens, targets, target_lens