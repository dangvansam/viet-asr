import os, time
import argparse
from typing import Tuple, Union
import pkbar

import torch
from torch.optim import AdamW, Adam
from torch.optim.optimizer import Optimizer

from loss import CrossEntropyLoss, SmoothCrossEntropyLoss, WarmupLR, SmoothCTCLoss

from models import asr_model

from utils import *
from config import load_config



def get_loss_factor(ctc_weight: float, att_weight: float) -> Tuple[float, float]:
    return ctc_weight, att_weight


def train_on_epoch(
            training_model: asr_model.ASR, 
            kbar: pkbar.Kbar,
            opt_func: Optimizer, 
            schedulers: WarmupLR,
            criterion_crossEntropy: Union[CrossEntropyLoss, SmoothCrossEntropyLoss], 
            criterion_ctc: SmoothCTCLoss,
            device: str = "cpu", 
            acc_steps: int = 4,
            batch_size: int = 16,
            *args, **kwargs
        ) -> float:

    global training_set_generator, model_config, dataloader_generation
    training_model.train(mode=True)

    k = 0   
    train_loss = 0
    encoder_loss_total = 0
    encoder_acc = 0
    num_samples = 0
    num_words = 0
    num_words_decoder = 0
    decoder_acc = 0

    use_decoder = model_config['model'].get('use_decoder', None)

    opt_func.zero_grad()

    for length_batch, waveforms, target_encoder, target_decoder, input_decoder, ctc_target, ctc_target_lengths, pseudo in training_set_generator:

        if type(waveforms) == list or length_batch % batch_size != 0:
            continue

        k += 1

        waveforms = waveforms.to(device)

        if use_decoder:
            tgt_mask = gen_mask(input_decoder.shape[1])
            input_decoder = input_decoder.to(device)
            tgt_mask = tgt_mask.to(device)
            encoder_outputs, decoder_output = training_model(waveforms, input_decoder, tgt_mask, gradient_mask= pseudo)

            target_decoder = target_decoder.to(device)
            ce_decoder_loss = criterion_crossEntropy(decoder_output, target_decoder)
            ce_encoder_factor = model_config.get("loss", dict()).get('en_weight', 0.4)
            ce_decoder_factor = model_config.get("loss", dict()).get('de_weight', 0.6)
        else:
            encoder_outputs, decoder_output = training_model(waveforms, gradient_mask= pseudo)

        # compute crossEntropy loss
        if dataloader_generation.use_ce_loss:
            target_encoder = target_encoder.to(device)
            encoder_outputs = encoder_outputs.permute(0,2,1)    # shape (B, vocab, SE)
            encoder_loss = criterion_crossEntropy(encoder_outputs, target_encoder)

            num_words += target_encoder.shape[1] * length_batch                    # total word in batch
            encoder_acc += (encoder_outputs.argmax(1) == target_encoder).sum().item()
            encoder_acc_step = encoder_acc / num_words
        
        if dataloader_generation.use_ctc_loss:
            encoder_outputs = encoder_outputs.log_softmax(2).permute(1, 0, 2)   # shape (SE, B, D)

            SE, B, D = encoder_outputs.shape
            ctc_input_lengths = B * [SE]
            ctc_input_lengths = torch.tensor(ctc_input_lengths)
            
            # compute ctc loss
            ctc_target, ctc_input_lengths, ctc_target_lengths = \
                ctc_target.to(device), ctc_input_lengths.to(device), ctc_target_lengths.to(device)
            encoder_loss = criterion_ctc(encoder_outputs, ctc_target, ctc_input_lengths, ctc_target_lengths)

            encoder_acc_step = 0

        # if torch.isnan(encoder_loss).any().item() == True or torch.isnan(encoder_loss).any().item() == True:
        #     continue

        if use_decoder:
            loss = (ce_encoder_factor * encoder_loss + ce_decoder_factor * ce_decoder_loss)/ acc_steps
        else:
            loss = encoder_loss / acc_steps

        loss.backward()

        torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm= 5.0, norm_type=2) 

        if k % acc_steps == 0:
            opt_func.step()

            if schedulers is not None:
                schedulers.step()

            opt_func.zero_grad()

        train_loss += loss.detach().item()
        encoder_loss_total += encoder_loss.detach().item()

        num_samples += 1
        loss_step = encoder_loss_total / num_samples

        if use_decoder:
            num_words_decoder += target_decoder.shape[1] * length_batch
            decoder_acc += (decoder_output.argmax(1) == target_decoder).sum().item()
            decoder_acc_step = decoder_acc / num_words_decoder
            kbar.update(k, values=[("en_loss", loss_step), ("en_acc", encoder_acc_step), 
                                ("de_acc", decoder_acc_step), ("lr_rate", opt_func.param_groups[0]["lr"]) ])
        else:
            kbar.update(k, values=[("en_loss", loss_step), ("en_acc", encoder_acc_step), ("lr_rate", opt_func.param_groups[0]["lr"]) ])

    return train_loss / len(training_set_generator)


def test(
        training_model: asr_model.ASR, 
        criterion_crossEntropy: Union[CrossEntropyLoss, SmoothCrossEntropyLoss],
        criterion_ctc: SmoothCTCLoss,
        device: str = "cpu", 
        batch_size: int = 16
    ) -> Tuple[float, float]:

    global model_config, dataloader_generation

    test_loss = 0
    # encoder
    num_words = 0
    encoder_acc = 0
    # decoder
    num_words_decoder = 0
    decoder_acc = 0
    val_de_acc = 0

    use_decoder = model_config['model'].get('use_decoder', None)

    training_model.train(mode=False)

    for length_batch, waveforms, target_encoder, target_decoder, input_decoder, ctc_target, ctc_target_lengths, pseudo in validate_set_generator:

        if type(waveforms) == list or length_batch % batch_size != 0:
            continue

        with torch.no_grad():
            waveforms = waveforms.to(device)

            if use_decoder:
                tgt_mask = gen_mask(input_decoder.shape[1])
                input_decoder = input_decoder.to(device)
                tgt_mask = tgt_mask.to(device)
                encoder_outputs, decoder_output = training_model(waveforms, input_decoder, tgt_mask)
            
                target_decoder = target_decoder.to(device)
                ce_decoder_loss = criterion_crossEntropy(decoder_output, target_decoder)
                ce_encoder_factor = model_config.get("loss", dict()).get('en_weight', 0.4)
                ce_decoder_factor = model_config.get("loss", dict()).get('de_weight', 0.6)
            else:
                encoder_outputs, _ = training_model(waveforms)
            
            # compute crossEntropy loss
            if dataloader_generation.use_ce_loss:
                target_encoder = target_encoder.to(device)
                encoder_outputs = encoder_outputs.permute(0,2,1)    # shape (B, vocab, SE)
                encoder_loss = criterion_crossEntropy(encoder_outputs, target_encoder)

                num_words += target_encoder.shape[1] * length_batch                         # total word in batch
                encoder_acc += (encoder_outputs.argmax(1) == target_encoder).sum().item()
            
            if dataloader_generation.use_ctc_loss:
                encoder_outputs = encoder_outputs.log_softmax(2).permute(1, 0, 2)   # shape (SE, B, D)

                SE, B, D = encoder_outputs.shape
                ctc_input_lengths = B * [SE]
                ctc_input_lengths = torch.tensor(ctc_input_lengths)
                
                # compute ctc loss
                ctc_target, ctc_input_lengths, ctc_target_lengths = \
                    ctc_target.to(device), ctc_input_lengths.to(device), ctc_target_lengths.to(device)
                encoder_loss = criterion_ctc(encoder_outputs, ctc_target, ctc_input_lengths, ctc_target_lengths)
                num_words = 1

            if use_decoder:
                loss = ce_encoder_factor * encoder_loss + ce_decoder_factor * ce_decoder_loss
            else:
                loss = encoder_loss

            test_loss += loss.detach().item()

            if use_decoder:
                num_words_decoder += target_decoder.shape[1] * length_batch                         # total word in batch
                decoder_acc += (decoder_output.argmax(1) == target_decoder).sum().item()
    
    valid_loss, val_en_acc = test_loss / len(validate_set_generator), encoder_acc / num_words

    if use_decoder:
        val_de_acc = decoder_acc / num_words_decoder
       
    return valid_loss, val_en_acc, val_de_acc



def train_model(
        vocab: int,
        training_model: asr_model.ASR, 
        num_epoch: int, 
        pretrained: bool,
        learning_rate: float, 
        device: str, 
        save_folder_path: str, 
        state_dict_path: str,
        num_snapshots: int,
        acc_steps: int,
        batch_size: int,
        weight_decay: float,
        freeze_decoder: bool = False,
        freeze_encoder: bool = False,
        *args, **kwargs

    ) -> Tuple[float, int]:

    global training_set_generator, validate_set_generator, epoch_unfreeze, unfreeze, ignore_fc_layer, use_adamw

    training_model = training_model.to(device)

    checkpoint_dict = dict()
    steps_per_epoch = int(len(training_set_generator) / acc_steps)
    learning_rate = float(learning_rate)

    if use_adamw:
        print("\n Using AdamW Optimizer !")
        opt_func = AdamW(training_model.parameters(), lr= learning_rate, betas=(0.9, 0.98), eps=1e-09, weight_decay= weight_decay)
    else:
        print("\n Using Adam Optimizer !")
        opt_func = Adam(training_model.parameters(), lr= learning_rate, betas=(0.9, 0.98), eps=1e-08, weight_decay= weight_decay)

    if pretrained:
        stage_dict = torch.load(state_dict_path, map_location= torch.device(device))
        
        if ignore_fc_layer:
            for name_layer in list(stage_dict.keys()):
                if "final_fc" in name_layer or "subword_embedding" in name_layer:
                    stage_dict.pop(name_layer)

        training_model.load_state_dict(stage_dict, strict= False)

        schedulers = torch.optim.lr_scheduler.OneCycleLR(
            opt_func, max_lr= learning_rate, steps_per_epoch= steps_per_epoch, epochs= num_epoch, pct_start= 0.01
        )
    else:
        schedulers = torch.optim.lr_scheduler.OneCycleLR(
            opt_func, max_lr= learning_rate, steps_per_epoch= steps_per_epoch, epochs= num_epoch
        )
        # initialize_weights(training_model)

    if freeze_decoder:
        print("Freeze decoder!")
        training_model.subword_embedding.requires_grad_(False)
        training_model.pos_decoder.requires_grad_(False)
        training_model.attention_decoder.requires_grad_(False)
    
    if freeze_encoder:
        print("Freeze encoder!")
        
        if training_model.ssl_training:
            training_model.encoder.requires_grad_(False)
        else:
            training_model.subsampling.requires_grad_(False)
            training_model.pos_encoder.requires_grad_(False)
            training_model.attention_encoder.requires_grad_(False)

    print("\nTraining model is on cuda: {}".format(next(training_model.parameters()).is_cuda))

    criterion_crossEntropy = SmoothCrossEntropyLoss(classes= vocab)
    criterion_ctc = SmoothCTCLoss(num_classes= vocab, blank= 0)

    for epoch in range(num_epoch):
        
        if unfreeze and epoch + 1 >= epoch_unfreeze:
            print("\nUnfreeze model at epoch {epoch}".format(epoch= epoch + 1))
            training_model.requires_grad_(True)
            unfreeze = False
        
        kbar = pkbar.Kbar(target=len(training_set_generator) + 1, epoch= epoch, num_epochs= num_epoch, width= 8, always_stateful= True)

        train_loss = train_on_epoch(training_model, kbar, opt_func, schedulers, 
                criterion_crossEntropy, criterion_ctc, device, acc_steps, batch_size)

        valid_loss, val_en_acc, val_de_acc = test(training_model, criterion_crossEntropy, criterion_ctc, device, batch_size)

        kbar.add(1, values=[("val_loss", valid_loss), ("val_en_acc", val_en_acc), ("val_de_acc", val_de_acc)])
        
        update_checkpoint(training_model, checkpoint_dict, save_folder_path, epoch, num_snapshots)
        
    return



def run(
        args_input: argparse.Namespace, 
        model_config: dict
    ):

    global vocab, ensemble_training

    training_model = asr_model.ASR(vocab, model_config['model'])

    train_params = {
        "save_folder_path": args_input.save_folder_path, 
        "device": args_input.device, 
        "pretrained": args_input.pretrained,
        "state_dict_path": args_input.pretrained_path,
        "learning_rate": args_input.learning_rate, 
        "num_epoch": args_input.num_epoch, 
        "freeze_decoder": args_input.freeze_decoder,
        "freeze_encoder": args_input.freeze_encoder,
        "num_snapshots": args_input.num_snapshots,
        "acc_steps": args_input.acc_steps,
        "batch_size": args_input.batch_size,
        "weight_decay": args_input.weight_decay
    }

    train_model(vocab, training_model, **train_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config model path')
    parser.add_argument('--save_folder_path', type=str, required=True, help= 'the checkpoint folder path')
    
    parser.add_argument('--device', type=str, default= "cpu", required=True, 
                            help= 'the training device, cpu or cuda')
    
    parser.add_argument('--pretrained_path', type=str, default= None, required=False, 
                            help= 'pretrained path, file .pt')
    
    parser.add_argument('--type_encoder_loss', type=str, default= "ce", required=True, 
                            help= 'type of encoder loss, ctc loss or ce loss')
    
    parser.add_argument('--learning_rate', type=float, default= 0.001, required= True,
                            help= 'the learning rate value')
    
    parser.add_argument('--weight_decay', type=float, default= 0.0001, required= True,
                            help= 'the weight decay value')
    
    parser.add_argument('--num_epoch', type=int, default= 100, required= True,
                            help= 'number epoch training')
    
    parser.add_argument('--num_snapshots', type=int, default= 10, required= True,
                            help= 'the number of the last saved checkpoint')
    
    parser.add_argument('--acc_steps', type=int, default= 32, required= True,
                            help="the accumulate gradients step")
    
    parser.add_argument('--batch_size', type=int, default= 8, required= True,
                            help= 'the batch size training')
    
    parser.add_argument('--epoch_unfreeze', type=int, default= 1, required= False,
                            help= 'the unfreeze epoch')

    parser.add_argument('--pretrained', action='store_true',
                            help= 'used when using pretrained')

    parser.add_argument('--freeze_decoder', action='store_true', 
                            help= 'freeze decoder layers')

    parser.add_argument('--freeze_encoder', action='store_true',
                            help= 'freeze encoder layers')

    parser.add_argument('--unfreeze', action='store_true',
                            help= 'unfreeze model when training')
                            
    parser.add_argument('--ignore_fc_layer', action='store_true',
                            help= 'ignore last fc layer when use pretrained')

    parser.add_argument('--use_adamw', action='store_true',
                            help= 'use adamw optimizer, use adam optimizer if not specify')

    parser.add_argument('--ensemble_training', action='store_true',
                            help= 'training ensemble ssl model')

    # parse args input
    args_input = parser.parse_args()

    config_path = args_input.config
    save_folder_path = args_input.save_folder_path
    epoch_unfreeze= args_input.epoch_unfreeze
    unfreeze = args_input.unfreeze
    ignore_fc_layer = args_input.ignore_fc_layer
    type_encoder_loss = args_input.type_encoder_loss
    use_adamw = args_input.use_adamw
    ensemble_training = args_input.ensemble_training

    # make save folder
    os.system("mkdir -p {}".format(save_folder_path))
    os.system("cp {} {}/model_config.yaml".format(config_path, save_folder_path))

    # make log dir
    log_dir = os.path.join(save_folder_path, "log")
    model_config = load_config.load_yaml(config_path)

    # make data loader
    dataloader_generation = Data_Generator(
                                    type_encoder_loss= type_encoder_loss, 
                                    batch_size= args_input.batch_size, 
                                    **model_config['asr_dataset']
                                )
    vocab, training_set_generator, validate_set_generator = \
        dataloader_generation.gen_data(**model_config['asr_dataset'])

    # train
    run(args_input, model_config)

