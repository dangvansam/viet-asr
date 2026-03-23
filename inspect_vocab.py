import sys
import torch
import warnings
warnings.filterwarnings('ignore')

from nemo.collections.asr.models import ASRModel

model_path = "/home/samdv/multitalker-asr/models/multitalker-parakeet-streaming-0.6b-v1.nemo"
model = ASRModel.restore_from(model_path, map_location='cpu')

if hasattr(model, 'tokenizer'):
    vocab_size = model.tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    for i in range(100, 150):
        print(f"ID {i}: {model.tokenizer.ids_to_text([i])}")
    print("...")
    # check if 'cực' or 'Vì' or 'Mỗi' is in the vocab mapping
    try:
        print("cực:", model.tokenizer.text_to_tokens("cực"))
        print("Mỗi:", model.tokenizer.text_to_tokens("Mỗi"))
        print("Vì", model.tokenizer.text_to_tokens("Vì"))
        print("hầu", model.tokenizer.text_to_tokens("hầu"))
    except Exception as e:
        print("Error", e)
