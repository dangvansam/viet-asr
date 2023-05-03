import os
import time
from gtts import gTTS
from playsound import playsound
from threading import Thread

def text2speech(text):
    tts = gTTS(text=text, lang='vi')
    tts.save('tmp.mp3')
    playsound('tmp.mp3')
    os.remove("tmp.mp3")

def text2speech_thread(text):
    th = Thread(target=text2speech, args=(text,), daemon=True)
    th.start()

if __name__=="__main__":
    text2speech_thread("xin chào")
    time.sleep(5)