import argparse
from vietasr.asr_task import ASRTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config model path')
    parser.add_argument('-f', '--test_meta_filepath', type=str, required=True, help='test meta file path')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parser.add_argument('-d', '--device', type=str, default="cpu", help='test on device')

    args = parser.parse_args()

    task = ASRTask(config=args.config)

    task.run_test(
        test_meta_filepath=args.test_meta_filepath,
        model_path=args.model_path,
        device=args.device
    )
