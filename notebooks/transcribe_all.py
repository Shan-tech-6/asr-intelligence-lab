import os
import whisper

model = whisper.load_model("base")

audio_folder = "audio"
output_folder = "transcripts"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        audio_path = os.path.join(audio_folder, file)
        result = model.transcribe(audio_path)

        output_file = os.path.join(output_folder, file.replace(".wav", ".txt"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Finished: {file}")

print("All audio files have been transcribed.")