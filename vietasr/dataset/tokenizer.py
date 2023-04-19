from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import torch
import sentencepiece as spm

class SentencepiecesTokenizer():
    def __init__(self, model: Union[Path, str]):
        self.model = str(model)
        self.sp = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line)

    def text2ids(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsIds(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))

    def ids2text(self, ids: List[int]) -> str:
        self._build_sentence_piece_processor()
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.DecodeIdsWithCheck(ids)
