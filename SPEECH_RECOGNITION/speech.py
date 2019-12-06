import os
import playsound
import speech_recognition as sp
from random import randrange
from gtts import gTTS

greetings_in = ["ciao","ciao python","ehy","buongiorno"]
greetings_out = ["hey","come va?","eccomi!","ciao daniel"]

def speak(text):
    tts =gTTS(text=text,lang='it')
    filename = "voce.mp3"
    if os.path.exists(filename):
        os.remove(filename)
    tts.save(filename)
    playsound.playsound(filename)

def get_audio():
    r = sp.Recognizer()
    with sp.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source)
        result = ""
        try:
            result = r.recognize_google(audio,language='it')
            print(result)
        except Exception as e:
            print("errore")
    return result

while True:
    text_in = get_audio()
    text_in = text_in.lower()
    text_out = text_in
    if text_in!=None and text_in!="":
        if text_in in greetings_in:
            ran = randrange(len(greetings_out))
            text_out = greetings_out[ran]
        speak(text_out)