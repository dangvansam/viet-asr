import math

from typing import List, Tuple
from torch import nn
from torch import Tensor

from models.pre_encoder import sinusoidal_positional_encoding, subsampling, gradient_masking, feature_extraction
from models.pre_decoder import embedding

from models import encoder, decoder
from utils import initialize_weights


class BaseModel(nn.Module):
    ''' The Seq2Seq ASR BaseModel

    '''

    def __init__(self,
                vocab_size: int,
                encoder_params: List[dict],
                decoder_params: List[dict],
                features_extractor_params: List[dict],
                num_mel_bins: int = 80,
                subsampling_channel: int = 256,
                subsampling_dropout: float = 0.3,
                type_encoder: str = "conformer",
                type_decoder: str = "transformer",
                sinusoidal_pos_enc_style: str = "concat",
                position_dropout: float = 0.2,
                use_gradient_mask: bool = False,
                use_last_dropout: bool = True,
                use_decoder: bool = False,
                *args, **kwargs
            ):
        
        super(BaseModel, self).__init__()

        encoder_param, decoder_param, features_extractor_params = encoder_params[0], decoder_params[0], features_extractor_params[0]
        self.use_gradient_mask = use_gradient_mask
        self.use_last_dropout = use_last_dropout
        self.use_decoder = use_decoder
        self.ssl_training = False

        self.feature_extractor = feature_extraction.Extractor(**features_extractor_params)
        
        if type_encoder == 'conformer':
            self.attention_encoder = encoder.conformer_ver2.Encoder(**encoder_param)

        elif type_encoder == 'transformer':
            self.attention_encoder = encoder.transformer.Encoder(**encoder_param)
        
        else:
            raise(ValueError, "Current version only support 'conformer', 'transformer' encoder !")

        hidden_dim = encoder_params[0]["d_model"]
        self.encoder_final_fc = nn.Linear(hidden_dim, vocab_size)
        initialize_weights(self.encoder_final_fc)

        if not self.ssl_training:
            self.subsampling = subsampling.Conv2dSubsampling_4(
                                    input_dim= num_mel_bins, 
                                    out_channels= subsampling_channel, 
                                    out_dim= hidden_dim, 
                                    input_dropout_p= subsampling_dropout
                                )

            self.pos_encoder = sinusoidal_positional_encoding.Embedd(
                                    hidden_dim, 
                                    position_dropout, 
                                    style= sinusoidal_pos_enc_style
                                )
            
            # init weight
            initialize_weights(self.subsampling)
            initialize_weights(self.pos_encoder)
            initialize_weights(self.attention_encoder)

        if use_gradient_mask:
            self.mask = gradient_masking.Mask_Hidden_Features(mask_dim= hidden_dim)
            initialize_weights(self.mask)
        
        if use_decoder:
            self.subword_embedding = embedding.SubWord_Embedding(vocab_size, hidden_dim)
            self.pos_decoder = sinusoidal_positional_encoding.Embedd(hidden_dim, position_dropout, style= sinusoidal_pos_enc_style)

            if type_decoder == 'transformer':
                self.attention_decoder = decoder.transformer.Decoder(**decoder_param)
            else:
                raise(ValueError, "Current version only support 'transformer' decoder !")

            self.decode_final_fc = nn.Linear(hidden_dim, vocab_size)
            self.last_dropout_decoder = nn.Dropout(p= 0.1)

            # init weight
            initialize_weights(self.subword_embedding)
            initialize_weights(self.pos_decoder)
            initialize_weights(self.attention_decoder)
            initialize_weights(self.decode_final_fc)
        


class ASR(BaseModel):

    def __init__(self,
                vocab_size: int,
                model_params: dict
            ):
        
        super(ASR, self).__init__(vocab_size, **model_params)

    def run_encoder(self, waveform: Tensor) -> Tensor:
        if not self.ssl_training:
            x = self.feature_extractor(waveform)
            x = self.subsampling(x)
            x = self.pos_encoder(x)
            x = self.attention_encoder(x)[1]
        else:
            B, C, T = waveform.shape
            waveform = waveform.reshape((B, T))
            x = self.encoder(waveform)[1]

        return x
    
    def run_decoder(self,
            memory_encode: Tensor,
            input_decoder: Tensor,
            tgt_mask: Tensor
        ) -> List[Tensor]:

        y = self.subword_embedding(input_decoder)
        y = self.pos_decoder(y)
        y, attn_output_weights = self.attention_decoder(y, memory_encode, tgt_mask= tgt_mask)
        y = self.decode_final_fc(y)
        y = self.last_dropout_decoder(y)
        return y, attn_output_weights

    def forward(self,
            waveform: Tensor = None, 
            input_decoder: Tensor = None, 
            tgt_mask: Tensor = None,
            gradient_mask: bool = False
        ) -> Tuple[Tensor, Tensor]: 

        ''' Forward ASR model

        Args:
            waveform: input waveform, (batch, channel, time)
            input_decoder: input transformer decoder, (B, SD)
            tgt_mask: the additive mask for the tgt sequence, (SD, SD) 
            
        Returns: 
            (tuple): tuple containing:

            encoder_outputs: output of encoder with shape (B, SE, vocab)
            decoder_outputs: output of decoder with shape (B, vocab, SD)
        
        Notes:
            B: batch_size
            SE: encoder sequence length
            SD: decoder sequence length

        '''
        x = None
        y = None

        if waveform is not None:
            
            if not self.ssl_training:
                memory_encode = self.feature_extractor(waveform)
                memory_encode = self.subsampling(memory_encode)
                memory_encode = self.pos_encoder(memory_encode)
                memory_encode = self.attention_encoder(memory_encode)[1]
            else:
                B, C, T = waveform.shape
                waveform = waveform.reshape((B, T))
                memory_encode = self.encoder(waveform)[1]

            # mask acoustic features
            if self.use_gradient_mask and gradient_mask:
                mask, mask_input = gradient_masking.make_mask(x)
                mask = mask.to(x.device)
                mask_input = mask_input.to(x.device)
                memory_encode = self.mask(memory_encode, mask, mask_input)

            x = self.encoder_final_fc(memory_encode)

        if input_decoder is not None and self.use_decoder:
            y = self.subword_embedding(input_decoder)
            y = self.pos_decoder(y)
            y = self.attention_decoder(y, memory_encode, tgt_mask= tgt_mask)[0]
            y = self.decode_final_fc(y)
            y = self.last_dropout_decoder(y)
            y = y.permute(0,2,1)

        return x, y