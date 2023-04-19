import os, sys, glob, json, argparse, ast
from tqdm import tqdm

sys.path.append(os.getcwd())
print(sys.path)

from torch import Tensor

import torch
import torchaudio

from execution_time import ExecutionTime

from vietasr.pre_encoder.transform import MelSpectrogram, AmplitudeToDB, RangeNorm


'''

python preprocessing/compute_global_cepstral_mean.py \
    --meta_data_folder dataset/meta_training/   \
    --save_json_path dataset/global_cmvn_infor.json \
    --devide cuda

'''


extime = ExecutionTime()


def save_json(data: dict, path: str) -> None:
    data = json.dumps(data, indent=4, ensure_ascii=False, sort_keys=False)
    with open(path, mode="w", encoding="utf8") as fp:
        fp.write(data)
    return



class Global_CMVN_Computer(torch.nn.Module):

    def __init__(
            self,
            num_mels: int = 80,
            stride: float = 0.015,
            window_size: float = 0.025,
            sample_rate: int = 16000,
            n_fft: int = 1024,
        ) -> None:

        super(Global_CMVN_Computer, self).__init__()

        self.hop_length = int( (window_size-stride) * sample_rate)
        self.num_mels = num_mels

        extractor = [
                MelSpectrogram(sample_rate = sample_rate, n_fft= n_fft, hop_length= self.hop_length, n_mels= self.num_mels),
                AmplitudeToDB(),
                RangeNorm()
            ]
        
        self.extractor = torch.nn.Sequential(*extractor)

    @extime.timeit
    def forward(
            self, 
            waveform: Tensor,
        ) -> Tensor:
        
        ''' Features Extraction foward
        
        Args:
            - waveform (Tensor): Batch of waveform inputs (batch, channel, time)
        Outputs:
            - Features (Tensor): Batch of Fbank features (B, D, SE)
        '''

        features = self.extractor(waveform)

        return features


def run(
        meta_data_folder: str,
        save_json_path: str,
        device: str = "cuda"
    ) -> None:

    global_computer = Global_CMVN_Computer()
    global_computer = global_computer.to(device)
    
    meta_paths = glob.glob(os.path.join(meta_data_folder, "*.txt"))

    wav_paths = list()

    for path in meta_paths:
        with open(path, encoding="utf8") as fp:
            for line in fp:
                line = line.strip()

                audio_path, label, dur = line.split("|")
                dur = ast.literal_eval(dur)

                if not os.path.isfile(audio_path) or dur < 1: 
                    continue

                wav_paths.append(audio_path)
    
    num_audio = len(wav_paths)
    total_frames = 0
    global_mean = 0

    for i in tqdm(range(num_audio)):
        if i == 100: break

        path = wav_paths[i]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.to(device)
        features = global_computer(waveform)
        total_frames += features.shape[2]

        if i == 0:
            sum_frame = features.sum(dim=2)
        else:
            sum_frame += features.sum(dim=2)

    global_mean = sum_frame[0] / total_frames
    global_mean = global_mean.detach().cpu().tolist()

    output = {"meta_data_folder": meta_data_folder, "total_frames": total_frames, "global_mean_value": global_mean, "num_audio": num_audio}
    save_json(output, save_json_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data_folder', type=str, required=True)
    parser.add_argument('--save_json_path', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)

    args = parser.parse_args()
    args = vars(args)
    run(**args)

    log_time = extime.logtime_data
    print("\n{}\n".format(log_time))
