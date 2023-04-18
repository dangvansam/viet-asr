import torch
from models.pre_encoder.feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
import numpy as np

fe = FeatureExtractor()

inputs = torch.randn((4, 32000))
inputs_lens = torch.LongTensor([16000, 8000, 12000, 32000])

feats, feat_lens = fe(inputs, inputs_lens)

print(feats)
print(feats.shape)
print(feat_lens.shape)

plt.imshow(feats[2].T)
plt.show()