
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
                        print(f"skipped long file, duration={dur}")
                        continue
                    if not os.path.exists(wav_filepath):
                        print(wav_filepath)
                        # exit()
                    if len(text.strip()) == 0:
                        continue
                    data.append((line[0], line[1]))
                    break
        logger.info(f"loaded {len(data)} samples")
        print(data[:10])
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
        
    def __call__(self, batch: Tuple[str, str]):
        inputs = [torchaudio.load(b[0])[0].squeeze(0) for b in batch]
        input_lens = torch.LongTensor([x.shape[0] for x in inputs])
        
        targets = [torch.LongTensor(self.tokenizer.text2ids(b[1])) for b in batch]
        target_lens = torch.LongTensor([len(x) for x in targets])
        
        inputs = pad_list(inputs, pad_value=0.0)
        targets = pad_list(targets, pad_value=0)
        
        return inputs, input_lens, targets, target_lens