import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import speech_recognition as sr
import pyttsx3 as tts
from ai_voice_assistant_pytorch.assistant import Assistant
recognizer = sr.Recognizer()

speaker = tts.init()
speaker.setProperty("rate", 150)
voices = speaker.getProperty("voices")
speaker.setProperty("voice", voices[1].id)
threshold = 0.75
assistant = Assistant(threshold)

bot_name = "Cy"

with sr.Microphone() as source:

    while True:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            print("Say something!")
            
            audio = recognizer.listen(source)

            text = recognizer.recognize_google(audio)
            text = text.lower()
            clean_text = re.sub(r"[^\w\s]", '', text)
            

            print(f"{bot_name} thinks you said {text}")

            # assistant
            response = assistant.respond(text)

            print(f"{bot_name}: {response}")

            speaker.say(response)
            speaker.runAndWait()

            if clean_text == "bye":
                speaker.stop()
                break

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google {e}")

