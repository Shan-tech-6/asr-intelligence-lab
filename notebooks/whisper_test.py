import whisper

model = whisper.load_model("base")

audio_path = "audio/audio.wav"

result = model.transcribe(audio_path)

print("\n--- TRANSCRIPTION ---\n")
print(result["text"])