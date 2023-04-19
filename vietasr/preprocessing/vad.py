
import argparse, glob, os, hashlib, sys
sys.path.append(os.getcwd())

from typing import Callable, List
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

import torch
import torchaudio

def write_txt(
        data: List[str],
        path: str
    ) -> None:

    with open(path, mode='w', encoding='utf8') as fp:
        fp.writelines(data)

def hash_str(string: str) -> int:
    return int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 10**10

def init_vad(USE_ONNX: bool = True) -> tuple:

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False,
                                onnx=USE_ONNX)

    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils

    return model, read_audio, get_speech_timestamps

def get_speech_in_audio(
        audio_path: str,
        model: Callable,
        read_audio: Callable,
        get_speech_timestamps: Callable,
        sampling_rate: int = 16000,
    ) -> list:

    bgin = datetime.now()
    wav = read_audio(audio_path, sampling_rate= sampling_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    speech_timestamps_seconds = [{"start": x['start']/sampling_rate, "end": x['end']/sampling_rate} for x in speech_timestamps]
    # pprint(speech_timestamps_seconds[:10])
    # print("get speech timestamps time: ", datetime.now() - bgin)
    return wav, speech_timestamps


def save_torch_audio(
        path: str,
        waveform: torch.Tensor,
        sample_rate: int
    ):
    torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)


def run(
        wav_folder_input: str,
        wav_folder_output: str,
        meta_output_path: str,
        min_dur: float = 1,
        max_dur: float = 16,
        sample_rate: int = 16000,
        hash_name: bool = False
    ):

    os.system(f"mkdir -p {wav_folder_output}")

    model, read_audio, get_speech_timestamps = init_vad()
    paths = glob.glob(os.path.join(wav_folder_input, "*.wav"))
    
    min_samples = min_dur * sample_rate
    max_samples = max_dur * sample_rate

    meta_data = list()
    total_dur = 0 

    for wav_path in tqdm(paths):
        wav_name = os.path.basename(wav_path)

        if hash_name:
            wav_name = hash_str(wav_name)
        else:
            wav_name = wav_name.split(".")[0]

        waveform, speech_timestamps = get_speech_in_audio(wav_path, model, read_audio, get_speech_timestamps)
        k = 0

        for infor in speech_timestamps:
            k += 1
            start = infor['start']
            end = infor['end']
            segment_sample = end - start
            segment_dur = segment_sample / sample_rate
            total_dur += segment_dur

            start_second = round(start/sample_rate, 3)
            end_second = round(end/sample_rate, 3)

            if segment_sample < min_samples or segment_sample > max_samples: continue

            segment_tensor = waveform[start:end]
            segment_path = os.path.join(wav_folder_output, f"{wav_name}_{start_second}_{end_second}.wav")

            save_torch_audio(segment_path, segment_tensor.unsqueeze(0), sample_rate)
            new_line = f"{segment_path}|UNK|{segment_dur}\n"
            meta_data.append(new_line)
        
        # break
    
    print(f"\nTotal duration: {total_dur}s")
    write_txt(meta_data, meta_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_folder_input', type=str, required=True)
    parser.add_argument('--wav_folder_output', type=str, required=True)
    parser.add_argument('--meta_output_path', type=str, required=True)
    
    # parse args input
    args_input = parser.parse_args()
    args_input = vars(args_input)
    run(**args_input)