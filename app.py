import numpy as np
import argparse
from src.preprocessing import load_audio, bandpass_filter, spectral_noise_reduction
from src.feature_extraction import extract_features
from src.model import compute_quality_score, classify_quality
from src.evaluation import visualize_audio, visualize_spectrogram

def analyze_audio(file_path):
    y, sr = load_audio(file_path)
    visualize_audio(y, sr, "Original Audio")
    visualize_spectrogram(y, sr, "Original Spectrogram")

    y_clean = spectral_noise_reduction(bandpass_filter(y, sr), sr)
    visualize_audio(y_clean, sr, "Cleaned Audio")
    visualize_spectrogram(y_clean, sr, "Cleaned Spectrogram")

    f = extract_features(y_clean, sr)
    score = compute_quality_score(f)
    label = classify_quality(score)

    print(f"Quality Score: {score:.2f}")
    print(f"Classification: {label}")
    return score, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Announcement Quality Analyzer")
    parser.add_argument("--file", type=str, required=True, help="Path to input audio file")
    args = parser.parse_args()
    analyze_audio(args.file)
