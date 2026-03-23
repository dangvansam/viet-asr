import sys
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    from nemo.collections.asr.models import ASRModel
except ImportError as e:
    sys.exit(1)

model_path = "/home/samdv/multitalker-asr/models/multitalker-parakeet-streaming-0.6b-v1.nemo"
model = ASRModel.restore_from(model_path, map_location='cpu')

if hasattr(model, 'tokenizer'):
    vi_text = "Vẫn chưa cảm thấy cân bằng hay"
    try:
        tokens = model.tokenizer.text_to_tokens(vi_text)
        ids = model.tokenizer.text_to_ids(vi_text)
        print(f"Text: {vi_text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
    except Exception as e:
        print(f"Tokenization failed: {e}")
    print("---")
    
    if hasattr(model.tokenizer, 'vocab_size'):
        print(f"Tokenizer vocab size: {model.tokenizer.vocab_size}")
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'blank_idx'):
        print(f"Decoder blank_idx: {model.decoder.blank_idx}")
    elif hasattr(model, 'joint') and hasattr(model.joint, 'blank_idx'):
        print(f"Joint blank_idx: {model.joint.blank_idx}")
        
    print(f"Joint vocabulary size: {model.joint.vocabulary_size if hasattr(model, 'joint') else 'Unknown'}")
