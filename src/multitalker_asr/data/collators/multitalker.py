from typing import Any, Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from .base import BaseCollator


class MultitalkerCollator(BaseCollator):
    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple:
        audios = [torch.from_numpy(item["audio"]) for item in batch]
        audio_lens = torch.tensor([item["audio_len"] for item in batch])

        if batch[0]["text_ids"]:
            text_ids = [torch.tensor(item["text_ids"]) for item in batch]
            text_lens = torch.tensor([len(ids) for ids in text_ids])
            padded_text = pad_sequence(text_ids, batch_first=True, padding_value=0)
        else:
            padded_text = [item["text"] for item in batch]
            text_lens = torch.tensor([len(t) for t in padded_text])

        spk_masks = [torch.from_numpy(item["spk_mask"]) for item in batch]
        bg_masks = [torch.from_numpy(item["bg_mask"]) for item in batch]

        padded_audio = pad_sequence(audios, batch_first=True)
        padded_spk_mask = pad_sequence(spk_masks, batch_first=True)
        padded_bg_mask = pad_sequence(bg_masks, batch_first=True)

        return (
            padded_audio,
            audio_lens,
            padded_text,
            text_lens,
            padded_spk_mask,
            padded_bg_mask,
        )
