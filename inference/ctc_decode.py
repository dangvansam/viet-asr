import os, sys, json

sys.path.append(os.getcwd())
print(sys.path)

import argparse

from execution_time import ExecutionTime

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from pyctcdecode import build_ctcdecoder
from pyctcdecode.decoder import BeamSearchDecoderCTC
from flashtext import KeywordProcessor

from inference.infer_utils import *

from config import load_config
from utils import write_txt


extime = ExecutionTime()


def save_json(data: dict, path: str) -> None:
    data = json.dumps(data, indent=4, ensure_ascii=False, sort_keys=False)
    with open(path, mode="w", encoding="utf8") as fp:
        fp.write(data)
    return



def build_pyctc_decoder(
            kenlm_model_path: str,
            alpha: float = 0.5,
            beta: float = 1.5,
            ctc_token_idx: int = 0,
        ) -> BeamSearchDecoderCTC:
    
    '''Build CTC Decoder

        Following open source: https://github.com/kensho-technologies/pyctcdecode

    '''
    global testset

    labels = [testset.bpe_model.IdToPiece(id) for id in range(testset.bpe_model.GetPieceSize())]
    labels[ctc_token_idx] = ""

    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path,
        alpha= alpha,  # tuned on a val set
        beta= beta,  # tuned on a val set
    )

    return decoder



def print_beamresults(
        beam_results: Tensor,
        labels: list,
        beam_scores: Tensor,
        search_pro: BpeProcessor
    ) -> None:

    for i, batch_i in enumerate(beam_results):
        print("\n")
        utt_beam_scores = beam_scores[i]
        for k, beams_i in enumerate(batch_i):
            beams_i = beams_i.tolist()
            beams_i_score = utt_beam_scores[k].tolist()
            beams_i_chars = convert_2_char(beams_i, labels)
            
            beams_i_text = "".join(beams_i_chars)
            beams_i_text = beams_i_text.split("_")[0]
            w_founded = search_pro.extract_keywords(beams_i_text)
            print("beam {}: {} , score: {}".format(k, w_founded, beams_i_score))


def build_kw_norm():
    keyword_processor = KeywordProcessor()
    kw_list = [ ("S Z N", "SẴN"), ("S ZN", "SẴN"),
                ("BI Z FLY", "BIZFLY"), ("BI Z", "BIZ"), ("Z ALO", "ZALO"),
                ("S Z C", "SJC"), ("S Z XI", "SJC"),
                ("LA Z ADA", "LAZADA"), ("A LÔ", "ALO")]
    
    for x, y in kw_list:
        keyword_processor.add_keyword(x, y)

    return keyword_processor


@extime.timeit
def test_bpe(
        device: str, 
        decoder: BeamSearchDecoderCTC,
        batch_size: int,
        beam_size: int,
        save_path: str
    ) -> None:

    test_transcript = list()
    test_set_generator = DataLoader(testset, batch_size, shuffle= False, num_workers= 5, collate_fn= my_collate_v2)
    keyword_processor = build_kw_norm()

    for waveforms, audio_paths, label, duration in test_set_generator:
        waveforms = waveforms.to(device)
        batch_transcript = list()
        
        with torch.no_grad():
            encoder_outputs = model.run_encoder(waveforms)
            encoder_outputs = model.encoder_final_fc(encoder_outputs)
            encoder_outputs = encoder_outputs.log_softmax(2)
            print(encoder_outputs.shape)

            for i, log_i in enumerate(encoder_outputs):
                log_i = log_i.cpu().detach().numpy()

                try:
                    output = decoder.decode_beams(log_i, beam_width= beam_size)
                    print("\n")
                
                    for x in output[:5]:
                        _text_ = keyword_processor.replace_keywords(x[0])
                        print("Beam {}: {}, lm score: {}".format(i, _text_, x[-1]))

                    predicted = output[0][0]
                    predicted = keyword_processor.replace_keywords(predicted)
                    batch_transcript.append(predicted)
                    
                except Exception as ex:
                    batch_transcript.append("")
    
        for i, prediction in enumerate(batch_transcript):
            line = "{path}|{predict}|{ground_truth}|{dur}\n".format(
                path= audio_paths[i],
                predict= prediction,
                ground_truth= label[i],
                dur= duration[i] 
            )
            test_transcript.append(line)
    
    write_txt(test_transcript, save_path)

    return 


@extime.timeit
def run(
        kenlm_rescore_path: str,
        device: str,
        beam_size: int,
        batch_size: int,
        save_folder: str,
        alpha: float = 0.5,
        beta: float = 1.5
    ) -> None:

    os.system("mkdir -p {}".format(save_folder))
    save_path = os.path.join(save_folder, "transcript_ctc_decode.txt")

    decoder = build_pyctc_decoder(kenlm_rescore_path, alpha, beta)
    test_bpe(device, decoder, batch_size, beam_size, save_path)

    save_log_path = os.path.join(save_folder, "time_log_ctc_decode.json")
    log_time = extime.logtime_data
    save_json(log_time, save_log_path)
    print(log_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, required=True, help= 'the inference config path')
    parser.add_argument('--wav_path_meta', type=str, required=True, help= 'the metadata test file')
    parser.add_argument('--checkpoint_folder', type=str, required=True, help= 'the checkpoint folder path')
    parser.add_argument('--model_state_dict_path', type=str, required=True, help= 'the model state dict path', default= "")
    parser.add_argument('--device', type=str, required=True, help='cuda or cpu')
    parser.add_argument('--type_model', type=str, required=False, default='seq2seq', help= 'the type of model to use')

    parser.add_argument('--kenlm_rescore_path', type=str, required=True, help= 'the kemlm path')
    parser.add_argument('--beam_size', type=int, required=True, default= 15, help= 'beam size decoding')
    parser.add_argument('--num_avg', type=int, required=True, default= 5, help= 'the number of checkpoint to avg')
    parser.add_argument('--batch_size', type=int, required=True, help= 'the batch size decoding')
    parser.add_argument('--save_folder', type=str, required=True, help= 'the output decode folder')
    parser.add_argument('--alpha', type=float, required=True, help= 'the weight of kenlm in decoding', default= 0.5)
    parser.add_argument('--beta', type=float, required=True, help= 'the weight of the length of sentence in decoding', default= 1.5)


    args_input = parser.parse_args()
    args_input = vars(args_input)

    args = load_config.load_yaml(args_input['config_path'])
    testset, vocab = build_testset_loader(args['asr_dataset']['bpe_model_path'], args_input['wav_path_meta'])

    model_args = {
        'vocab': vocab,
        'checkpoint_folder': args_input['checkpoint_folder'],
        'model_params': args['model'],
        'device': args_input['device'],
        'model_state_dict_path': args_input['model_state_dict_path'],
        'num_avg': args_input['num_avg'],
        'type_model': args_input['type_model']
    }

    model = load_model(**model_args)

    run_args = {
        'kenlm_rescore_path': args_input['kenlm_rescore_path'],
        'device': args_input['device'],
        'beam_size': args_input['beam_size'],
        'batch_size': args_input['batch_size'],
        'save_folder': args_input['save_folder'],
        'alpha': args_input['alpha'],
        'beta': args_input['beta']
    }

    run(**run_args)