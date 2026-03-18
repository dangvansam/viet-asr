from multitalker_asr.model import MultitalkerASRModel
from multitalker_asr.config import ModelConfig
import os


def test_model_loading():
    print("Testing model loading...")
    model_cfg = ModelConfig(device="cpu", cuda_id=-1)
    model = MultitalkerASRModel(model_cfg)
    model.load_models()
    if model.asr_model and model.diar_model:
        print("Models loaded successfully!")
    else:
        print("Model loading failed.")


if __name__ == "__main__":
    test_model_loading()
