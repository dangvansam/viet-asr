import torch
from nemo.collections.asr.models import EncDecMultiTalkerRNNTBPEModel
from loguru import logger

model_path = "/home/samdv/multitalker-asr/checkpoints/multitalker-vietnamese-scratch.nemo"
model = EncDecMultiTalkerRNNTBPEModel.restore_from(model_path, map_location="cpu")
model.eval()

# Try passing a dummy audio
audio = torch.randn(1, 64000) # [B, T]
audio_len = torch.tensor([64000])

with torch.no_grad():
    encoder_output, encoder_len = model.encoder(audio_signal=audio, length=audio_len)
    predictions, hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output, encoder_len, return_hypotheses=True
    )

print("Predictions text:", hypotheses[0].text)
