import torch
from models.model import Model
from config.load_config import load_yaml

config = load_yaml("config/conformer.yaml")

model = Model(2000, **config["model"])
print(model)

inputs = torch.randn((4, 16000))
input_lens = torch.LongTensor([16000, 16000, 16000, 16000])
input_decoder = torch.randint(0, 2000, (4, 5))
mask = torch.ones_like(input_decoder).bool()

print(mask)
out = model(inputs, input_lens)