import argparse
from vietasr.asr_task import ASRTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config model path')
    parser.add_argument('-o', '--output_dir', type=str, help='the checkpoint folder path')
    parser.add_argument('-d', '--device', type=str, default="cpu", help='device is cpu or cuda')
    args = parser.parse_args()

    task = ASRTask(
        config=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    task.run_train()
