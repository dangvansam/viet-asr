import sys
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    from nemo.collections.asr.models import ASRModel
except ImportError as e:
    print(f"Failed to import NeMo: {e}")
    sys.exit(1)

print("Loading ASR model...")
try:
    model_path = "/home/samdv/multitalker-asr/models/multitalker-parakeet-streaming-0.6b-v1.nemo"
    import os
    if os.path.exists(model_path):
        model = ASRModel.restore_from(model_path, map_location='cpu')
    else:
        model = ASRModel.from_pretrained("nvidia/multitalker-parakeet-streaming-0.6b-v1", map_location='cpu')
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
        vocab = model.decoder.vocabulary
        print(f"Vocabulary size: {len(vocab)}")
        print(f"First 50 tokens: {vocab[:50]}")
        print(f"Last 50 tokens: {vocab[-50:]}")
    else:
        print("Model does not have decoder.vocabulary")
        
    print(f"Tokenizer type: {type(model.tokenizer) if hasattr(model, 'tokenizer') else 'None'}")
    
    vietnamese_chars = ['à', 'á', 'ã', 'ạ', 'ả', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'đ', 'è']
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        for char in vietnamese_chars[:5]:
            try:
                tokens = model.tokenizer.text_to_tokens(char)
                print(f"Char '{char}' -> Tokens: {tokens}")
            except Exception as e:
                print(f"Tokenizing '{char}' failed: {e}")
    else:
        if hasattr(model.decoder, 'vocabulary'):
            print("Model uses character based vocabulary, checking if Vietnamese characters exist...")
            for char in vietnamese_chars:
                if char not in model.decoder.vocabulary:
                    print(f"Missing character: {char}")
except Exception as e:
    print(f"Error inspecting model: {e}")
