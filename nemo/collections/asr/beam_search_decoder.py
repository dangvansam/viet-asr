# Copyright (c) 2019 NVIDIA Corporation
# Requires Baidu's CTC decoders from
# https://github.com/PaddlePaddle/DeepSpeech/decoders/swig

import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core import DeviceType
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs
from nemo.utils.helpers import get_cuda_device
from pyctcdecode import build_ctcdecoder

class BeamSearchDecoderWithLM(NonTrainableNM):
    """Neural Module that does CTC beam search with a n-gram language model.

    It takes a batch of log_probabilities. Note the bigger the batch, the
    better as proccessing is parallelized. Outputs a list of size batch_size.
    Each element in the list is a list of size beam_search, and each element
    in that list is a tuple of (final_log_prob, hyp_string).

    Args:
        vocab (list): List of characters that can be output by the ASR model. For Jasper, this is the 28 character set
            {a-z '}. The CTC blank symbol is automatically added later for models using ctc.
        beam_width (int): Size of beams to keep and expand upon. Larger beams result in more accurate but slower
            predictions
        alpha (float): The amount of importance to place on the n-gram language model. Larger alpha means more
            importance on the LM and less importance on the acoustic model (Jasper).
        beta (float): A penalty term given to longer word sequences. Larger beta will result in shorter sequences.
        lm_path (str): Path to n-gram language model
        num_cpus (int): Number of cpus to use
        cutoff_prob (float): Cutoff probability in vocabulary pruning, default 1.0, no pruning
        cutoff_top_n (int): Cutoff number in pruning, only top cutoff_top_n characters with highest probs in
            vocabulary will be used in beam search, default 40.
        input_tensor (bool): Set to True if you intend to pass pytorch Tensors, set to False if you intend to pass
            numpy arrays.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "log_probs": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),}),
            # "log_probs_length": NeuralType({0: AxisType(BatchTag)}),
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "log_probs_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        predictions:
            NeuralType(None)
        """
        # return {"predictions": NeuralType(VoidType())}
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(
        self,
        lm_path,
        vocab,
        beam_width,
        alpha,
        beta,
        num_cpus,
        cutoff_prob=1.0,
        cutoff_top_n=40,
        input_tensor=True
    ):
        super().__init__()
        # Override the default placement from neural factory and set placement/device to be CPU.
        self._placement = DeviceType.CPU
        self._device = get_cuda_device(self._placement)

        if self._factory.world_size > 1:
            raise ValueError("BeamSearchDecoderWithLM does not run in distributed mode")

        self.decoder = build_ctcdecoder(
            vocab,
            kenlm_model_path=lm_path,  # either .arpa or .bin file
            alpha=alpha,  # tuned on a val set
            beta=beta,  # tuned on a val set
        )
        self.vocab = vocab
        self.beam_width = beam_width
        self.num_cpus = num_cpus
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.input_tensor = input_tensor

    def forward(self, log_probs, log_probs_length):
        assert log_probs.size(0) == 1, f"log_probs.shape={log_probs.shape}, batch size must be 1"
        probs = torch.exp(log_probs[0]).cpu().numpy()
        text = self.decoder.decode(
            logits=probs,
            beam_width=self.beam_width
        )
        return text