import torch
from vietasr.model.asr_model import ASRModel
from config.load_config import load_yaml

config = load_yaml("config/conformer.yaml")

model = ASRModel(2000, **config["model"])
print(model)

inputs = torch.randn((4, 16000))
input_lens = torch.LongTensor([16000, 10000, 12000, 8000])
target = torch.randint(0, 2000, (4, 4))
target_lens = torch.LongTensor([1,2,3,4])

out = model(inputs, input_lens, target, target_lens)
print(out)