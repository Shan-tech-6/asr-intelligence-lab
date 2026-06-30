import sounddevice as sd
from scipy.io.wavfile import write

fs=44100  # Sample rate
seconds=5  # Duration of recording  

print("speak now...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write("audio\\audio.wav", fs, recording)  # Save as WAV file
print("Recording complete. Audio saved as 'audio\\audio.wav'.")