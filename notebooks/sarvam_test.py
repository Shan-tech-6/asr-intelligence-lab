import os
from sarvamai import SarvamAI

client = SarvamAI(
    api_subscription_key=""
)

audio_folder = "audio"
output_folder = "transcripts/sarvam"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        print(f"Processing: {file}")

        with open(os.path.join(audio_folder, file), "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe",
                language_code="en-IN",
                input_audio_codec="wav"
            )

        output_file = os.path.join(
            output_folder,
            file.replace(".wav", ".txt")
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.transcript)

        print(f"Saved: {output_file}")

print("All audio files transcribed successfully!")