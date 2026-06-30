import os
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")

audio_folder = "audio"
output_folder = "transcripts/faster_whisper"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        audio_path = os.path.join(audio_folder, file)

        segments, info = model.transcribe(audio_path)

        text = ""
        for segment in segments:
            text += segment.text + " "

        output_file = os.path.join(output_folder, file.replace(".wav", ".txt"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text.strip())

        print(f"Finished: {file}")

print("All files transcribed using Faster-Whisper.")