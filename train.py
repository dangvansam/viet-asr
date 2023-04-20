import argparse
from vietasr.bin.asr_task import ASRTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config model path')
    parser.add_argument('-o', '--output_dir', type=str, help= 'the checkpoint folder path')
    args = parser.parse_args()

    task = ASRTask(config=args.config, output_dir=args.output_dir)
    task.run_train()
