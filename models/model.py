from typing import List, Tuple

from torch import Tensor, nn

# from models.encoder.conformer import ConformerEncoder
from models.encoder.conformer.encoder import ConformerEncoder 
from models.encoder.transformer import TransformerEncoder
from models.pre_encoder.feature_extraction import Extractor
from models.pre_encoder.positional_encoding import PositionalEncoding
from models.pre_encoder.subsampling import Conv2dSubsampling_4
from models.decoder.transformer import TransformerDecoder
from models.pre_decoder.embedding import WordEmbedding
from utils import initialize_weights


class Model(nn.Module):
    '''
    The Seq2Seq ASR BaseModel
    '''
    def __init__(
        self,
        vocab_size: int,
        encoder_params: dict,
        decoder_params: dict,
        features_extractor_params: dict,
        num_mel_bins: int = 80,
        subsampling_channel: int = 256,
        subsampling_dropout: float = 0.1,
        type_encoder: str = "conformer",
        type_decoder: str = "transformer",
        sinusoidal_pos_enc_style: str = "concat",
        position_dropout: float = 0.1,
        use_last_dropout: bool = True,
    ):

        super(Model, self).__init__()

        # encoder_param, decoder_param, features_extractor_params = encoder_params[0], decoder_params[0], features_extractor_params[0]
        hidden_dim = encoder_params["d_model"]
        
        # Encoder
        self.use_last_dropout = use_last_dropout
        self.feature_extractor = Extractor(**features_extractor_params)

        # self.subsampling = Conv2dSubsampling_4(
        #     input_dim=num_mel_bins,
        #     out_channels=subsampling_channel,
        #     out_dim=hidden_dim,
        #     input_dropout_p=subsampling_dropout
        # )

        # self.pos_encoder = PositionalEncoding(
        #     hidden_dim,
        #     position_dropout,
        #     style=sinusoidal_pos_enc_style
        # )

        if type_encoder == 'conformer':
            # self.encoder = ConformerEncoder(**encoder_params)
            self.encoder = ConformerEncoder(
                input_dim=encoder_params["d_model"],
                encoder_dim=encoder_params["d_model"],
                num_layers=encoder_params["n_layers"],
                num_attention_heads=encoder_params["nhead"]
            )
        elif type_encoder == 'transformer':
            self.encoder = TransformerEncoder(**encoder_params)
        else:
            raise(
                ValueError, "Current version only support 'conformer', 'transformer' encoder !")
        
        self.encoder_final_fc = nn.Linear(hidden_dim, vocab_size)
        
        # Decoder
        self.embed_decoder = WordEmbedding(vocab_size, hidden_dim)
        self.pos_decoder = PositionalEncoding(
            hidden_dim, position_dropout, style=sinusoidal_pos_enc_style)

        if type_decoder == 'transformer':
            self.decoder = TransformerDecoder(**decoder_params)
        else:
            raise(ValueError, "Current version only support 'transformer' decoder !")

        self.decode_final_fc = nn.Linear(hidden_dim, vocab_size)
        self.last_dropout_decoder = nn.Dropout(p=0.1)

        # init weight
        # initialize_weights(self.subsampling)
        # initialize_weights(self.pos_encoder)
        initialize_weights(self.encoder)
        initialize_weights(self.encoder_final_fc)
        initialize_weights(self.embed_decoder)
        initialize_weights(self.pos_decoder)
        initialize_weights(self.decoder)
        initialize_weights(self.decode_final_fc)

    def forward_encoder(self, waveform: Tensor, lengths: Tensor) -> Tensor:
        x = self.feature_extractor(waveform)
        # x = self.subsampling(x)
        # x = self.pos_encoder(x)
        x, x_lens = self.encoder(waveform, lengths)
        return x, x_lens

    def forward_decoder(
        self,
        encoder_out: Tensor,
        encoder_out_lens: Tensor,
        target: Tensor,
        target_lens: Tensor
    ) -> List[Tensor]:
        y = self.embed_decoder(target)
        y = self.pos_decoder(y)
        y, attn_output_weights = self.decoder(y, encoder_out, tgt_mask=target_lens)
        y = self.decode_final_fc(y)
        y = self.last_dropout_decoder(y)
        return y, attn_output_weights

    def forward(
        self,
        input: Tensor,
        input_lens: Tensor,
        target: Tensor = None,
        target_lens: Tensor = None,
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

        encoder_out, encoder_out_lens = self.forward_encoder(input, input_lens)
        x = self.encoder_final_fc(encoder_out)

        if target is not None:
            y = self.forward_decoder(encoder_out, target, target_lens)
            y = y.permute(0, 2, 1)

        return x, y
