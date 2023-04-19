import os, json, math, pickle, glob, ast
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data.dataloader import DataLoader

# from preprocessing import data
# from preprocessing.pre_utils import Train_Loading

def pad_list(xs: List[torch.Tensor], pad_value: float, max_len: int = 0) -> torch.Tensor:
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(max_len, max(x.size(0) for x in xs))
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

def pad_waveform(
        sequence_waveform: List[torch.Tensor],
        min_audio_seconds: int = 1,
        sample_rate: int = 16000,
        padding_value: float = 0.0
    ) -> List[torch.Tensor]:

    '''Padding list of 2D Tensor
    '''

    output = list()

    min_length = int(min_audio_seconds * sample_rate)
    wav_lengths = [x.shape[1] for x in sequence_waveform]
    max_length = max(min_length, max(wav_lengths))

    for tensor in sequence_waveform:
        tensor_length = tensor.shape[1]
        pad_length = max_length - tensor_length
        tensor = pad(tensor, (0, pad_length), value= padding_value)
        output.append(tensor)
    
    return output


def pad_label(
            sequence_label_tensor: List[torch.Tensor],
            padding_value: int = 0,
        ) -> Tuple[List[Tensor]]:

    '''Padding list of 1D Tensor
    '''

    output = list()

    label_lengths = [len(x) for x in sequence_label_tensor]
    max_length = max(label_lengths)

    for tensor in sequence_label_tensor:
        num_pad = max_length - len(tensor)
        tensor = pad(tensor, (0, num_pad), value= padding_value)
        output.append(tensor)
    
    return output


def load_json(path: str) -> dict:
    with open(path, encoding="utf8") as fp:
        data = json.load(fp)
    return data

def save_json(data: dict, path: str) -> None:
    data = json.dumps(data, indent=4, ensure_ascii=False, sort_keys=False)
    with open(path, mode="w", encoding="utf8") as fp:
        fp.write(data)
    return

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def save_pickle(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return

def write_txt(
        data: List[str],
        path: str
    ) -> None:

    with open(path, mode='w', encoding='utf8') as fp:
        fp.writelines(data)


def update_checkpoint(
        training_model: torch.nn.Module,
        checkpoint_dict: dict,
        save_folder_path: str,
        epoch: int,
        num_snapshots: int
    ) -> None:

    epoch += 1

    save_path = os.path.join(save_folder_path, "epoch_{}_state_dict.pt".format(epoch))
    checkpoint_dict[epoch] = save_path
    torch.save(training_model.state_dict(), save_path)
    print('\n\nSaved model at epoch {}\n'.format(epoch))

    delta = epoch - num_snapshots

    if delta in checkpoint_dict:
        old_checkpoint_path = checkpoint_dict[delta]
        os.system("rm {}".format(old_checkpoint_path))
        print('Removed old model checkpoint at epoch {}\n'.format(delta))


            
            
def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def gen_mask(sz: int) -> Tensor:
    x = torch.zeros(sz, sz)
    
    for i in range(sz):
        if i == sz: continue
        x[i][i+1:] = float('-inf')
    return x


def my_mix_collate(data):
    data, pseudo_label = data[0]
    data[0] = pseudo_label
    return data


# def gen_data_loader(
#         batch_size: int,
#         model_config: dict,
#         type_model: str = "transducer|wavlm|None",
#     ) -> Tuple[int, int, DataLoader, DataLoader]:

#     ground_truth_base_dataset = data.Base_Dataset(**model_config['asr_dataset'])

#     ground_truth_train = data.ASR_Dataset(type_dataset= "train", base_dataset= ground_truth_base_dataset, batch_size= batch_size)
#     training_set_generator = DataLoader(ground_truth_train, batch_size, **model_config['data_loader_train'], collate_fn= collate_fn, pin_memory=False)

#     ground_truth_validate = data.ASR_Dataset(type_dataset= "dev", base_dataset= ground_truth_base_dataset)
#     validate_set_generator = DataLoader(ground_truth_validate, batch_size, **model_config['data_loader_valid'], collate_fn= collate_fn, pin_memory=False)

#     ctc_vocab = ground_truth_base_dataset.ctc_vocab
#     decoder_vocab = ground_truth_base_dataset.decoder_vocab

#     if 'asr_pseudo_dataset' in model_config:
#         pseudo_base_dataset = data.Base_Dataset(**model_config['asr_pseudo_dataset'])
#         pseudo_train = data.ASR_Dataset(type_dataset= "train", base_dataset= pseudo_base_dataset)
#         pseudo_train_generator = DataLoader(pseudo_train, batch_size, **model_config['data_loader_train'], collate_fn= collate_fn, pin_memory=False)

#         training_set = data.Mix_Dataset(training_set_generator, pseudo_train_generator)
#         training_set_generator = DataLoader(training_set, batch_size= 1, **model_config['data_loader_mix'], collate_fn=my_mix_collate, pin_memory=False)

#     return ctc_vocab, decoder_vocab, training_set_generator, validate_set_generator


def filter_audio(
        word_alignments: dict,
        max_dur: float = 15.0,
        hop_length_seconds: float = 0.01,
        min_frames: int = 2
    ) -> None:

    def parse_word_infor(
            word: str,
            start: float,
            end: float
        ) -> tuple:
        return word, start, end

    def convert_length(length: int) -> int:
        new_length = math.floor( (length -3)/2 + 1 )
        new_length = math.floor( (new_length - 3)/2 + 1 )
        return new_length

    remove_list = list()

    for wav_path in word_alignments:
        segments = word_alignments[wav_path]

        for word_infor in segments:
            word, start, end = parse_word_infor(**word_infor)
            
            if word == "SIL": continue
            
            start = int(start/hop_length_seconds)
            end = int(end/hop_length_seconds)

            if start == 0:
                start_word_index = 0
            else:
                start_word_index = convert_length(start)
            
            end_word_index = convert_length(end)

            if end_word_index - start_word_index < min_frames:
                remove_list.append(wav_path)
                break

        wav_dur = segments[-1]['end']
        if wav_dur > max_dur:
            remove_list.append(wav_path)
    
    remove_list = set(remove_list)
    for x in remove_list:
        word_alignments.pop(x)

    print("Filted {num_wav} from data!".format(num_wav= len(remove_list)))

class Data_Generator:
    
    def __init__(
            self,
            bpe_model_path: str,
            type_model: str = "seq2seq",
            type_encoder_loss: str = "ce",
            batch_size: int = 16,
            *args, **kwargs
        ) -> None:

        self.type_model = type_model
        self.type_encoder_loss = type_encoder_loss
        self.bpe_model_path = bpe_model_path
        self.batch_size = batch_size
        self.use_ce_loss = True
        self.use_ctc_loss = False
    
        if type_model == "seq2seq":
            self.collate_fn = self.seq2seq_collate

        elif type_model == "transducer":
            self.collate_fn = self.transducer_collate
        
        if type_encoder_loss == 'ctc':
            self.use_ce_loss = False
            self.use_ctc_loss = True
    
    def seq2seq_collate(self, batch):
        waveform_list = list()
        target_encoder = list()

        input_decoder = list()
        target_decoder = list()

        ctc_target = list()
        ctc_target_lengths = list()

        for waveform, label, pad_id, input_decoder_id, output_decoder_id, pseudo in batch:
            # waveform
            waveform_list.append(waveform)

            # encoder label
            if label is not None:
                target_encoder.append(label)

            # decoder label
            input_decoder.append(input_decoder_id)
            target_decoder.append(output_decoder_id)

            # ctc label
            ctc_target.append(output_decoder_id[:-1])
            ctc_target_lengths.append(output_decoder_id.shape[0] - 1)
        
        length_batch = len(waveform_list)
        
        # make waveforms batch
        waveforms = pad_waveform(waveform_list)
        waveforms = torch.stack(waveforms, dim= 0)

        # make label encoder
        if target_encoder:
            target_encoder = pad_label(target_encoder, padding_value= pad_id)
            target_encoder = torch.stack(target_encoder, dim= 0)

        # make input decoder batch
        input_decoder = pad_label(input_decoder)
        input_decoder = torch.stack(input_decoder, dim= 0)

        # make output decoder batch
        target_decoder = pad_label(target_decoder, padding_value= pad_id)
        target_decoder = torch.stack(target_decoder, dim= 0)

        # make input ctc
        ctc_target = torch.cat(ctc_target, dim= 0)
        ctc_target_lengths = torch.tensor(ctc_target_lengths)

        return length_batch, waveforms, target_encoder, target_decoder, input_decoder, ctc_target, ctc_target_lengths, pseudo
    
    def transducer_collate(self, batch):
        pseudo_label = False
        
        waveform_list = list()
        target_encoder = list()

        input_predictor = list()
        target_predictor = list()

        target_transducer = list()
        target_transducer_lengths = list()

        for waveform, label, pad_id, input_decoder_id, output_decoder_id in batch:

            waveform_list.append(waveform)

            if label is not None:
                target_encoder.append(label)

            input_predictor.append(input_decoder_id)
            target_predictor.append(output_decoder_id)

            target_transducer.append(output_decoder_id[:-1])
            target_transducer_lengths.append(input_decoder_id.shape[0] - 1)

        length_batch = len(waveform_list)

        # make waveforms
        waveforms = pad_waveform(waveform_list)
        waveforms = torch.stack(waveforms, dim= 0)

        # make target encoder batch
        if target_encoder:
            target_encoder = pad_label(target_encoder, padding_value= pad_id)
            target_encoder = torch.stack(target_encoder, dim= 0)

        # make input predictor
        input_predictor = pad_label(input_predictor)
        input_predictor = torch.stack(input_predictor, dim= 0)

        # make target predictor
        target_predictor = pad_label(target_predictor, padding_value= pad_id)
        target_predictor = torch.stack(target_predictor, dim= 0)

        # make target transducer
        target_transducer = pad_label(target_transducer, padding_value= pad_id)
        target_transducer = torch.stack(target_transducer, dim= 0)

        target_transducer_lengths = torch.tensor(target_transducer_lengths)

        return [pseudo_label, length_batch, waveforms, target_encoder, input_predictor, target_predictor, target_transducer, target_transducer_lengths]

    def gen_data(self,
            meta_folder: str,
            word_alignments_path: str,
            meta_folder_pseudo: str = None,
            *args, **kwargs
        ):

        if self.use_ce_loss:
            word_alignments = load_json(word_alignments_path)
            filter_audio(word_alignments)

            ground_truth_train = data.ASR_Dataset_Alignments(word_alignments, self.bpe_model_path, type_dataset='train')
            ground_truth_valid = data.ASR_Dataset_Alignments(word_alignments, self.bpe_model_path, type_dataset='valid')
        
        elif self.use_ctc_loss:
            train_loader = Train_Loading(meta_folder, self.bpe_model_path)
            meta_dictionary = train_loader.load_meta()

            ground_truth_train = data.ASR_Dataset(
                                    meta_dictionary, 
                                    self.bpe_model_path, 
                                    type_dataset='train', 
                                    noise_folder_path= kwargs['noise_folder_path'],
                                    rir_folder_path= kwargs['rir_folder_path'],
                                    on_fly_augt= kwargs['on_fly_augt']
                                )

            # training set generator
            training_set_generator = DataLoader(
                                ground_truth_train, 
                                self.batch_size, 
                                shuffle= True, 
                                num_workers= 5, 
                                collate_fn= self.collate_fn, 
                                pin_memory=False
                            )
            
            if meta_folder_pseudo is not None:
                train_loader_pseudo = Train_Loading(meta_folder, self.bpe_model_path)
                meta_dictionary_pseudo = train_loader_pseudo.load_meta()

                ground_truth_train_pseudo = data.ASR_Dataset(
                                        meta_dictionary_pseudo, 
                                        self.bpe_model_path, 
                                        type_dataset='train_pseudo', 
                                        noise_folder_path= kwargs['noise_folder_path'],
                                        rir_folder_path= kwargs['rir_folder_path'],
                                        on_fly_augt= kwargs['on_fly_augt']
                                    )

                # training set pseudo generator
                training_set_generator_pseudo = DataLoader(
                                ground_truth_train_pseudo, 
                                self.batch_size, 
                                shuffle= True, 
                                num_workers= 5, 
                                collate_fn= self.collate_fn, 
                                pin_memory=False
                            )

                ground_truth = data.Mix_Dataset(training_set_generator, training_set_generator_pseudo)
                training_set_generator = DataLoader(
                    ground_truth, 
                    self.batch_size, 
                    shuffle= True, 
                    num_workers= 5, 
                    collate_fn= self.collate_fn, 
                    pin_memory=False
                )

            ground_truth_valid = data.ASR_Dataset(
                                    meta_dictionary, 
                                    self.bpe_model_path, 
                                    type_dataset='valid',
                                    noise_folder_path= kwargs['noise_folder_path'],
                                    rir_folder_path= kwargs['rir_folder_path'],
                                    on_fly_augt= kwargs['on_fly_augt']
                                )


        # validate set generator
        validate_set_generator = DataLoader(
                                ground_truth_valid, 
                                self.batch_size, 
                                shuffle= False, 
                                num_workers= 5, 
                                collate_fn= self.collate_fn, 
                                pin_memory=False
                            )
        # get vocab
        vocab = ground_truth_train.encoder_vocab

        return vocab, training_set_generator, validate_set_generator
