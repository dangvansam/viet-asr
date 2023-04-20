from vietasr.dataset.tokenizer import SentencepiecesTokenizer

tokenizer = SentencepiecesTokenizer("data/bpe_2000/bpe.model")
print(tokenizer.get_vocab())
print(tokenizer.get_vocab_size())