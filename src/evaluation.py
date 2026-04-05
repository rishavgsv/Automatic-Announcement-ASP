import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(y, sr, title="Waveform"):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_spectrogram(y, sr, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
