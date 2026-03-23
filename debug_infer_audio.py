import torch
import soundfile as sf
import numpy as np
from nemo.collections.asr.models import EncDecMultiTalkerRNNTBPEModel

model_path = "checkpoints/multitalker-vietnamese-scratch.nemo"
model = EncDecMultiTalkerRNNTBPEModel.restore_from(
    model_path, map_location="cpu")
model.eval()

# Create dummy audio
dummy_audio = np.random.randn(16000 * 3)  # 3 seconds
sf.write("dummy.wav", dummy_audio, 16000)

with torch.no_grad():
    predictions = model.transcribe(["dummy.wav"])

print("Predictions:", predictions)
