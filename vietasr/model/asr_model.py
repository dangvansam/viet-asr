from typing import List, Tuple

import torch
from torch import Tensor, nn

# from models.encoder.conformer import ConformerEncoder
from vietasr.model.encoder.conformer.encoder import ConformerEncoder
from vietasr.model.encoder.transformer import TransformerEncoder
from vietasr.model.layers.feature_extraction import FeatureExtractor
from vietasr.model.layers.positional_encoding import PositionalEncoding
from vietasr.model.layers.subsampling import Conv2dSubsampling_4
from vietasr.model.decoder.transformer import TransformerDecoder
from vietasr.model.layers.embedding import WordEmbedding
from vietasr.model.layers.label_smoothing_loss import LabelSmoothingLoss
from vietasr.model.layers.ctc import CTC
from vietasr.model.model_utils import make_pad_mask, initialize_weights


class ASRModel(nn.Module):
    '''
    ASRModel
    '''

    def __init__(
        self,
        vocab_size: int,
        encoder_params: dict,
        decoder_params: dict,
        features_extractor_params: dict,
        num_mel_bins: int = 80,
        subsampling_channel: int = 256,
        subsampling_dropout: float = 0.0,
        type_encoder: str = "conformer",
        type_decoder: str = "transformer",
        sinusoidal_pos_enc_style: str = "concat",
        position_dropout: float = 0.0,
        ctc_weight: float = 0.5,
        ctc_linear_dropout: float = 0.0,
        label_smoothing_weight: float = 0.1,
        label_smoothing_normalize_length: bool = True,
        blank_id: int = 0,
        pad_id: int = -1
    ):
        super(ASRModel, self).__init__()
        hidden_dim = encoder_params["d_model"]
        self.ctc_weight = ctc_weight
        self.blank_id = blank_id
        self.padding_id = pad_id
        
        self.feature_extractor = FeatureExtractor(**features_extractor_params)

        # Encoder
        if type_encoder == 'conformer':
            # self.encoder = ConformerEncoder(**encoder_params)
            self.subsampling = None
            self.pos_encoder = None
            self.encoder = ConformerEncoder(
                input_dim=num_mel_bins,
                encoder_dim=encoder_params["d_model"],
                num_layers=encoder_params["n_layers"],
                num_attention_heads=encoder_params["nhead"],
                input_dropout_p=0,
                feed_forward_dropout_p=0,
                attention_dropout_p=0,
                conv_dropout_p=0
            )
        elif type_encoder == 'transformer':
            self.subsampling = Conv2dSubsampling_4(
                input_dim=num_mel_bins,
                out_channels=subsampling_channel,
                out_dim=hidden_dim,
                input_dropout_p=subsampling_dropout
            )
            self.pos_encoder = PositionalEncoding(
                hidden_dim,
                position_dropout,
                style=sinusoidal_pos_enc_style
            )
            self.encoder = TransformerEncoder(**encoder_params)
        else:
            raise (
                ValueError, "Current version only support 'conformer', 'transformer' encoder !")

        self.ctc = CTC(
            odim=vocab_size,
            encoder_output_size=hidden_dim,
            dropout_rate=ctc_linear_dropout,
            blank_id=self.blank_id
        )

        # Decoder
        self.embed_decoder = WordEmbedding(vocab_size, hidden_dim, self.padding_id)
        self.pos_decoder = PositionalEncoding(
            hidden_dim, position_dropout, style=sinusoidal_pos_enc_style)

        if type_decoder == 'transformer':
            self.decoder = TransformerDecoder(**decoder_params)
        else:
            raise (ValueError, "Current version only support 'transformer' decoder !")

        self.last_dropout_decoder = nn.Dropout(p=0.1)
        self.decode_final_fc = nn.Linear(hidden_dim, vocab_size)
        
        self.label_smoothing_loss = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=self.padding_id,
            smoothing=label_smoothing_weight,
            normalize_length=label_smoothing_normalize_length
        )
        # init weight
        if self.subsampling is not None:
            initialize_weights(self.subsampling)
        if self.pos_encoder is not None:
            initialize_weights(self.pos_encoder)
        initialize_weights(self.encoder)
        initialize_weights(self.embed_decoder)
        initialize_weights(self.pos_decoder)
        initialize_weights(self.decoder)
        initialize_weights(self.decode_final_fc)

    def forward_encoder(self, inputs: Tensor, lengths: Tensor=None) -> Tuple[Tensor, Tensor]:
        x, xlens = self.feature_extractor(inputs, lengths)
        if self.subsampling is not None:
            x = self.subsampling(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x, xlens = self.encoder(x, xlens)
        return x, xlens

    def forward_decoder(
        self,
        encoder_out: Tensor,
        encoder_out_lens: Tensor,
        target: Tensor,
        target_lens: Tensor
    ) -> List[Tensor]:
        y = self.embed_decoder(target)
        y = self.pos_decoder(y)
        y = self.decoder(y, encoder_out, make_pad_mask(target_lens), make_pad_mask(encoder_out_lens))
        y = self.last_dropout_decoder(y)
        y = self.decode_final_fc(y)
        return y

    def forward(
        self,
        input: Tensor,
        input_lens: Tensor,
        target: Tensor = None,
        target_lens: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        ''' Forward ASR model
        Args:
            input: input waveform, (batch, time)
            input: input length, (batch,)
            target: input transformer decoder, (batch, seq_len)
            target_lens: target length, (batch,) 
        '''

        encoder_out, encoder_out_lens = self.forward_encoder(input, input_lens)
        loss = self.ctc(encoder_out, encoder_out_lens, target, target_lens)
        ctc_loss = loss
        
        decoder_out = None
        decoder_loss = None
        if target is not None:
            decoder_out = self.forward_decoder(encoder_out, encoder_out_lens, target, target_lens)
            decoder_loss = self.label_smoothing_loss(decoder_out, target)
            loss = ctc_loss * self.ctc_weight + decoder_loss * (1-self.ctc_weight)

        result = {
            "loss": loss,
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
            "ctc_loss": ctc_loss,
            "decoder_out": decoder_out,
            "decoder_out_lens": target_lens,
            "decoder_loss": decoder_loss
        }
        return result
    
    def get_predicts(self, encoder_out: Tensor, encoder_out_lens: Tensor)-> List[List[int]]:
        with torch.no_grad():
            ctc_outputs = self.ctc.argmax(encoder_out)
            predicts = []
            for i in range(ctc_outputs.size(0)):
                ctc_out = ctc_outputs[i][: encoder_out_lens[i]].tolist()
                ctc_out = [i for i in ctc_out if i not in [self.blank_id, self.padding_id]]
                predicts.append(ctc_out)
        return predicts
    
    def get_labels(self, target: Tensor, target_lens: Tensor)-> List[List[int]]:
        with torch.no_grad():
            labels = []
            for i in range(target.size(0)):
                t = target[i][: target_lens[i]].tolist()
                labels.append(t)
        return labels
