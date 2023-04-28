import argparse
from vietasr.asr_task import ASRTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config model path')
    parser.add_argument('-f', '--test_meta_filepath', type=str, required=True, help='test meta file path')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size for ctc beamseach decoder, 1 mean greedy decode')
    parser.add_argument('-l', '--kenlm_path', type=str, help='kenlm model for ctc beamseach decoder')
    parser.add_argument('--kenlm_alpha', type=float, default=0.2, help='kenlm alpha for ctc beamseach decoder')
    parser.add_argument('--kenlm_beta', type=float, default=1.5, help='kenlm beta for ctc beamseach decoder')
    parser.add_argument('-d', '--device', type=str, default="cpu", help='test on device')

    args = parser.parse_args()

    task = ASRTask(config=args.config, device=args.device)
    
    task.load_checkpoint(args.model_path)

    if args.beam_size > 1:
        task.setup_beamsearch(
            kenlm_path=args.kenlm_path,
            kenlm_alpha=args.kenlm_alpha,
            kenlm_beta=args.kenlm_beta,
            beam_size=args.beam_size
        )

    task.run_test(test_meta_filepath=args.test_meta_filepath)
