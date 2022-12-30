import os
import sys
from loguru import logger
import torch
import numpy as np
import librosa
import soundfile as sf
from ruamel.yaml import YAML
import nemo
import nemo.collections.asr as nemo_asr
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.collections.asr.helpers import post_process_predictions, post_process_transcripts


class AudioDataLayer(DataLayerNM):
    @property
    def output_ports(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
            torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = np.reshape(signal, [1, -1])
        self.signal_shape = np.expand_dims(
            self.signal.size, 0).astype(np.int64)
        self.output = True

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self


class VietASR:
    def __init__(
        self,
        config_file: str,
        encoder_checkpoint: str,
        decoder_checkpoint: str,
        device: str="cpu",
        lm_path: str=None,
        beam_width: int=20,
        lm_alpha: float=0.5,
        lm_beta: float=1.5
    ):

        assert os.path.exists(config_file), f"config file not found: {config_file}"
        assert os.path.exists(encoder_checkpoint), f"encoder checkpoint not found: {encoder_checkpoint}"
        assert os.path.exists(decoder_checkpoint), f"decoder checkpoint not found: {decoder_checkpoint}"
        
        logger.info(f"Init VietASR with params:")
        logger.info(f"========================")
        logger.info(f"+ config: {config_file}")
        logger.info(f"+ encoder_checkpoint: {encoder_checkpoint}")
        logger.info(f"+ decoder_checkpoint: {decoder_checkpoint}")
        logger.info(f"+ lm_path: {lm_path}")
        logger.info(f"+ lm_alpha: {lm_alpha}")
        logger.info(f"+ lm_beta: {lm_beta}")
        logger.info(f"+ device: {device}")
        logger.info(f"========================")

        yaml = YAML(typ="safe")
        with open(config_file, encoding="utf-8") as f:
            model_definition = yaml.load(f)

        model_definition['AudioToMelSpectrogramPreprocessor']['dither'] = 0
        model_definition['AudioToMelSpectrogramPreprocessor']['pad_to'] = 0

        if device == "gpu" and torch.cuda.is_available():
            device = nemo.core.DeviceType.GPU
        else:
            device = nemo.core.DeviceType.CPU

        neural_factory = nemo.core.NeuralModuleFactory(placement=device)

        data_layer = AudioDataLayer(
            sample_rate=model_definition['AudioToMelSpectrogramPreprocessor']['sample_rate'])

        data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            **model_definition['AudioToMelSpectrogramPreprocessor'])

        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **model_definition['JasperEncoder'])

        jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=model_definition['JasperEncoder']['jasper'][-1]['filters'],
            num_classes=len(model_definition['labels']))

        # greedy_decoder = nemo_asr.GreedyCTCDecoder()

        labels = model_definition['labels']
        # use_beamsearch = False

        if lm_path and os.path.exists(lm_path):
            try:
                import kenlm  # check kenlm
                logger.info(f"use beamsearch decoder with language model")
            except:
                logger.error(
                    "kenlm python bindings are not installed. Most likely you want to install it using: "
                    "pip install https://github.com/kpu/kenlm/archive/master.zip"
                )
                lm_path = None

        if lm_path is None:
            logger.info(f"use beamsearch decoder without language model")

        beamsearch_decoder = nemo_asr.BeamSearchDecoderWithLM(
            vocab=labels,
            beam_width=beam_width,
            alpha=lm_alpha,
            beta=lm_beta,
            lm_path=lm_path,
            num_cpus=max(1, os.cpu_count())
        )

        # load pre-trained model
        jasper_encoder.restore_from(encoder_checkpoint)
        jasper_decoder.restore_from(decoder_checkpoint)

        # process audio signal
        audio_signal, audio_signal_len = data_layer()
        processed_signal, processed_signal_len = data_preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len
        )
        # foward encoder
        encoded, encoded_len = jasper_encoder(
            audio_signal=processed_signal,
            length=processed_signal_len
        )
        # foward decoder
        log_probs = jasper_decoder(encoder_output=encoded)

        # beamsearch decode
        beam_predictions = beamsearch_decoder(log_probs=log_probs, log_probs_length=encoded_len)
        infer_tensors = [beam_predictions]

        self.data_layer = data_layer
        self.neural_factory = neural_factory
        self.infer_tensors = infer_tensors

    def transcribe(self, audio_signal: np.array):
        self.data_layer.set_signal(audio_signal)
        evaluated_tensors = self.neural_factory.infer(tensors=self.infer_tensors, verbose=False)
        hypotheses = evaluated_tensors[0][0]
        return hypotheses


if __name__=="__main__":

    input_dir = sys.argv[1]
    assert os.path.exists(input_dir), f"{input_dir} is not found, try again!"
    
    logger.info(f"transcribe audio file in : {input_dir}")

    config = 'configs/quartznet12x1_vi.yaml'
    encoder_checkpoint = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
    decoder_checkpoint = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
    lm_path = 'models/language_model/3-gram-lm.binary'

    vietasr = VietASR(
        config_file=config,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
        lm_path=lm_path,
        beam_width=100
    )

    for f in os.listdir(input_dir):
        if not f.endswith(".wav") and not f.endswith(".mp3"):
            logger.error(f"{f} format is not supported")
            continue
        logger.info("==============================")
        # audio_signal, sr = sf.read(os.path.join(input_dir, f)) // faster but cant load mp3 or 8k sample rate file
        audio_signal, sr = librosa.load(os.path.join(input_dir, f), sr=16000)
        if len(audio_signal) > 10 * sr:
            logger.info("audio file too long, skipped")
            continue
        transcript = vietasr.transcribe(audio_signal)
        logger.success(f"filename: {f}")
        logger.success(f"transcript: {transcript}")
        break
