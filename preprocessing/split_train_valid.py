import argparse
import random
import os

def split_meta_file(input_filepath:str, output_dir: str, valid_ratio: float=0.1):
    assert  valid_ratio < 1 and valid_ratio >= 0, valid_ratio
    lines = open(input_filepath, "r", encoding="utf-8").readlines()
    random.shuffle(lines)
    train_file = open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8")
    valid_file = open(os.path.join(output_dir, "valid.txt"), "w", encoding="utf-8")
    
    valid_file.writelines(lines[: int(len(lines) * valid_ratio)])
    train_file.writelines(lines[int(len(lines) * valid_ratio) :])
    print(f"split done, saved to {output_dir}/[train.txt, valid.txt]")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    split_meta_file(args.input_filepath, args.output_dir)