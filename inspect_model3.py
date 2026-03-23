import sys
import torch
import warnings
warnings.filterwarnings('ignore')

from nemo.collections.asr.models import ASRModel

model_path = "/home/samdv/multitalker-asr/models/multitalker-parakeet-streaming-0.6b-v1.nemo"
model = ASRModel.restore_from(model_path, map_location='cpu')

print(f"Token 0: {model.tokenizer.ids_to_text([0])}")
print(f"Token 1: {model.tokenizer.ids_to_text([1])}")
print(f"Token 2: {model.tokenizer.ids_to_text([2])}")
print(f"Token 1024 (blank?): {model.tokenizer.ids_to_text([1024]) if hasattr(model.tokenizer, 'ids_to_text') else 'N/A'}")
