import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# -----------------------------
# LOAD AUDIO
# -----------------------------
def load_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.normalize(y)
    return y, sr


# -----------------------------
# BANDPASS FILTER (Speech range)
# -----------------------------
def bandpass_filter(y, sr, lowcut=300.0, highcut=3400.0, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)

    return y_filtered


# -----------------------------
# LIGHT NOISE REDUCTION ✅ (FINAL FIX)
# -----------------------------
def spectral_noise_reduction(y, sr):
    """
    Lightweight denoising:
    - Keeps speech intact
    - Reduces minor noise
    - Avoids distortion
    """

    # Step 1: Moving average smoothing
    window_size = 5
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='same')

    # Step 2: Small noise floor removal
    threshold = 0.02
    y_clean = np.where(np.abs(y_smooth) > threshold, y_smooth, 0)

    # Step 3: Normalize again
    y_clean = librosa.util.normalize(y_clean)

    return y_clean


# -----------------------------
# SAVE AUDIO
# -----------------------------
def save_audio(y, sr, out_path):
    sf.write(out_path, y, sr)


# -----------------------------
# CLI TEST (optional)
# -----------------------------
if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]

    y, sr = load_audio(file_path)
    y = bandpass_filter(y, sr)
    y = spectral_noise_reduction(y, sr)

    save_audio(y, sr, "data/processed/cleaned.wav")