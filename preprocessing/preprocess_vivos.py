import os
import argparse
import soundfile as sf
from tqdm import tqdm
from typing import List, Tuple

def load_vivos(directory: str, subset: str="") -> List[Tuple[str, str]]:
    print(f"Loading data from: {directory}")
    if not os.path.exists(directory):
        print(f"dataset: {directory} is not exist!")
    label_path = os.path.join(directory, subset, 'prompts.txt')
    metadata = open(label_path, 'r', encoding='utf-8').readlines()
    prog_bar = tqdm(range(len(metadata)))
    data = []
    for idx in prog_bar:
        line = metadata[idx].strip()
        line = line.split(' ', 1)
        speaker = line[0].split("_")[0]
        wav_path = os.path.join(directory, subset, 'waves', speaker, f"{line[0]}.wav")
        text = line[1].lower()
        data.append([wav_path, text])
    return data

def create_meta_file(data: List[Tuple[str, str]], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for x in tqdm(data):
            if len(x) == 2:
                audio, sr = sf.read(x[0])
                duration = str(len(audio) / sr)
                x.append(duration)
            f.write(f"{'|'.join(x)}\n")
    print(f"saved to: {output_file}")
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--subset', default="train", type=str)
    parser.add_argument('--save_filepath', type=str, required=True)

    args = parser.parse_args()
    
    data = load_vivos(args.dataset_dir, args.subset)
    create_meta_file(data, args.save_filepath)