import os
import numpy as np

def generate_noise(duration=5, sr=16000):
    noise = np.random.normal(0, 1, sr * duration)
    noise = noise / np.max(np.abs(noise))  # normalize
    return noise
def ensure_path(path):
    """Create directory if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def normalize_score(score):
    return np.clip(score, 0, 100)
