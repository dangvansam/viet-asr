import os
import time
from gtts import gTTS
# from playsound import playsound
import pygame
from threading import Thread

pygame.mixer.init()

def text2speech(text):
    tts = gTTS(text=text, lang='vi')
    tts.save('tmp.mp3')
    # playsound('tmp.mp3')
    #initialize the mixer module
    #Load a music file for playback
    pygame.mixer.music.load("../Records/welcome.mp3")
    #Start the playback of the music stream
    pygame.mixer.music.play()

    os.remove("tmp.mp3")

def text2speech_thread(text):
    th = Thread(target=text2speech, args=(text,), daemon=True)
    th.start()

if __name__=="__main__":
    text2speech_thread("xin chào")
    time.sleep(5)