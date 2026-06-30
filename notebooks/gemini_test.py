from google import genai

client = genai.Client(api_key="")

print("Uploading audio...")

uploaded_file = client.files.upload(
    file="audio/audio0.wav"
)

print("Upload complete.")

interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=[
        {
            "type": "text",
            "text": "Generate a transcript of the speech."
        },
        {
            "type": "audio",
            "uri": uploaded_file.uri,
            "mime_type": uploaded_file.mime_type
        }
    ]
)

print(interaction.output_text)