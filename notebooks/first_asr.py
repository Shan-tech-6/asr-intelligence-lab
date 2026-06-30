import speech_recognition as sr
r=sr.Recognizer()

with sr.AudioFile("audio\\audio.wav") as source:
    audio=r.record(source)
    text=r.recognize_google(audio)
    print("Transcription: ")
    print(text)