import sys
import argparse
import pyaudio
import wave
import time
import keyboard
from loguru import logger
sys.path.append(".")
from vietasr.asr_task import ASRTask
from dist.raspberry_pi.text2speech import text2speech_thread

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='config path')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size for ctc beamseach decoder, 1 mean greedy decode')
    parser.add_argument('-l', '--kenlm_path', type=str, help='kenlm model for ctc beamseach decoder')
    parser.add_argument('-v', '--word_vocab_path', type=str, default="data/word_vocab.txt", help='vocab file for ctc beamseach decoder')
    parser.add_argument('--kenlm_alpha', type=float, default=0.2, help='kenlm alpha for ctc beamseach decoder')
    parser.add_argument('--kenlm_be ta', type=float, default=1.5, help='kenlm beta for ctc beamseach decoder')
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

    # Define the audio format
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    # Open the microphone
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=1024)
    except:
        logger.error("Cannot open microphone, check microphone is connected to Pi. Exited!")
        exit()

    # Define a function to start recording
    def start_recording():
        logger.info("start record")
        # Start recording
        global is_recording
        is_recording = True
        start_time = time.time()

        # Write the audio to a file
        wf = wave.open("output.wav", "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        while is_recording:
            frames = stream.read(1024)
            wf.writeframes(frames)

        # Close the file
        wf.close()

    # Define a function to stop recording
    def stop_recording():
        logger.info("stop record")
        # Stop recording
        global is_recording
        is_recording = False

    # Main loop
    while True:
        # Check if the keyboard button is pressed
        if keyboard.is_pressed("space"):
            if is_recording:
                stop_recording()
                text = task.transcribe("output.wav")
                text2speech_thread(text)
                logger.success(f"Text: {text}")
            else:
                start_recording()

        # Sleep for 100 ms
        time.sleep(0.1)