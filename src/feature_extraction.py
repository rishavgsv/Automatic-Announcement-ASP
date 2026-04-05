import librosa
import numpy as np

def extract_features(y, sr):
    """Extract audio features relevant to speech quality."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    features = {
        'mfcc_mean': np.mean(mfcc, axis=1),
        'spec_centroid': spec_centroid,
        'zcr': zcr,
        'rms': rms,
        'spec_bw': spec_bw
    }
    return features
