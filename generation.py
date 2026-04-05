import os
import random
import numpy as np
import librosa
import soundfile as sf
import csv
import time
from gtts import gTTS

# -----------------------------
# CONFIG
# -----------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LABEL_FILE = "data/labels.csv"
NOISE_FILE = "data/noise.wav"
SAMPLE_RATE = 16000

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# -----------------------------
# TEXT GENERATION
# -----------------------------
def generate_text():
    trains = ["12345 Express", "Rajdhani Express", "Shatabdi Express"]
    actions = ["arriving", "departing", "delayed"]
    platforms = ["platform 1", "platform 2", "platform 3", "platform 4"]

    return f"Attention please, train {random.choice(trains)} is {random.choice(actions)} on {random.choice(platforms)}."


# -----------------------------
# TEXT → SPEECH (FIXED)
# -----------------------------
def text_to_speech(text, output_path, retries=3):
    for attempt in range(retries):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
            return
        except Exception as e:
            print(f"⚠️ TTS failed (attempt {attempt+1}): {e}")
            time.sleep(2)

    raise Exception("❌ gTTS failed after retries")


# -----------------------------
# NOISE GENERATION
# -----------------------------
def generate_noise(duration=5, sr=16000):
    noise = np.random.normal(0, 1, sr * duration)
    return noise / (np.max(np.abs(noise)) + 1e-6)


def load_noise(noise_file):
    if os.path.exists(noise_file):
        try:
            noise_audio, _ = librosa.load(noise_file, sr=SAMPLE_RATE)
            print("✅ Loaded real noise file")
            return noise_audio
        except Exception as e:
            print(f"⚠️ Error loading noise file: {e}")

    print("⚠️ Using synthetic noise")
    return generate_noise()


# -----------------------------
# ADD NOISE USING SNR
# -----------------------------
def add_noise_snr(clean, noise, snr_db):
    min_len = min(len(clean), len(noise))
    clean = clean[:min_len]
    noise = noise[:min_len]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power + 1e-6))

    noisy = clean + scale * noise
    return noisy


# -----------------------------
# MAIN DATASET GENERATION
# -----------------------------
def generate_dataset(num_samples=30):  # keep small first
    print("🚀 Generating dataset...")

    noise_audio = load_noise(NOISE_FILE)

    with open(LABEL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "snr", "label", "duration", "text"])

        for i in range(num_samples):
            try:
                # STEP 1: TEXT
                text = generate_text()

                # STEP 2: TTS
                raw_path = os.path.join(RAW_DIR, f"clean_{i}.mp3")
                text_to_speech(text, raw_path)

                # STEP 3: LOAD AUDIO
                clean_audio, sr = librosa.load(raw_path, sr=SAMPLE_RATE)

                # -----------------------------
                # AUGMENTATION (FIXED)
                # -----------------------------
                clean_audio = librosa.effects.time_stretch(
                    y=clean_audio,
                    rate=random.uniform(0.9, 1.1)
                )

                clean_audio = librosa.effects.pitch_shift(
                    y=clean_audio,
                    sr=sr,
                    n_steps=random.randint(-2, 2)
                )

                # Volume variation
                clean_audio *= random.uniform(0.5, 1.2)

                # -----------------------------
                # ADD NOISE
                # -----------------------------
                snr = random.randint(5, 20)
                noisy_audio = add_noise_snr(clean_audio, noise_audio, snr)

                # Normalize safely
                noisy_audio = noisy_audio / (np.max(np.abs(noisy_audio)) + 1e-6)

                # -----------------------------
                # SAVE AUDIO
                # -----------------------------
                file_path = os.path.join(PROCESSED_DIR, f"sample_{i}.wav")
                sf.write(file_path, noisy_audio, sr)

                # -----------------------------
                # LABELING
                # -----------------------------
                if snr >= 15:
                    label = "Good"
                elif snr >= 10:
                    label = "Moderate"
                else:
                    label = "Poor"

                duration = len(noisy_audio) / sr

                writer.writerow([file_path, snr, label, duration, text])

                print(f"✅ Sample {i} → {label} (SNR={snr})")

            except Exception as e:
                print(f"❌ Error in sample {i}: {e}")

    print("🎉 Dataset generation completed!")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    generate_dataset(30)