import argparse
from utils import load_yaml
from vietasr.bin.trainer import ASRTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config model path')
    parser.add_argument('-o', '--output_dir', type=str, help= 'the checkpoint folder path')
    args = parser.parse_args()
    config = load_yaml(args.config)
    trainer = ASRTrainer(config=config, output_dir=args.output_dir)
    trainer.run()
