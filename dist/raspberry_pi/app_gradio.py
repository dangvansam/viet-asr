import sys
import argparse
import logging
from loguru import logger
import gradio as gr
sys.path.append(".")
from vietasr.asr_task import ASRTask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config path')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size for ctc beamseach decoder, 1 mean greedy decode')
    parser.add_argument('-l', '--kenlm_path', type=str, help='kenlm model for ctc beamseach decoder')
    parser.add_argument('-v', '--word_vocab_path', type=str, default="data/word_vocab.txt", help='vocab file for ctc beamseach decoder')
    parser.add_argument('--kenlm_alpha', type=float, default=0.2, help='kenlm alpha for ctc beamseach decoder')
    parser.add_argument('--kenlm_beta', type=float, default=1.5, help='kenlm beta for ctc beamseach decoder')
    parser.add_argument('-d', '--device', type=str, default="cpu", help='test on device')

    args = parser.parse_args()

    task = ASRTask(config=args.config, device=args.device)
    task.load_checkpoint(args.model_path)

    if args.beam_size > 1:
        task.setup_beamsearch(
            kenlm_path=args.kenlm_path,
            word_vocab_path=args.word_vocab_path,
            kenlm_alpha=args.kenlm_alpha,
            kenlm_beta=args.kenlm_beta,
            beam_size=args.beam_size
        )

    def transcribe(audio_uploaded, audio_recorded):
        if audio_uploaded:
            audio = audio_uploaded
        if audio_recorded:
            audio = audio_recorded
        text = task.transcribe(audio)
        return text
     
    app = gr.Interface(
        title="Vietnamese Speech to Text",
        fn=transcribe, 
        inputs=[
            gr.Audio(source="upload", type="filepath"),
            gr.Audio(source="microphone", type="filepath")
        ], 
        outputs="text",
        examples=[
            ["audio_samples/1.wav", None],
            ["audio_samples/2.wav", None],
            ["audio_samples/3.wav", None],
            ["audio_samples/4.wav", None],
            ["audio_samples/5.wav", None],
            ["audio_samples/035-00000063-00000742.wav", None],
            ["audio_samples/035-00000758-00001333.wav", None],
            ["audio_samples/VIVOSDEV01_R002.wav", None],
            ["audio_samples/VIVOSDEV15_020.wav", None],
        ]
    )

    app.launch(share=True, inbrowser=True, debug=True)  
