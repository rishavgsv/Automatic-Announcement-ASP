import numpy as np
import librosa
import re
import joblib

# =============================
# LOAD WHISPER MODEL
# =============================
try:
    import whisper
    whisper_model = whisper.load_model("base")
except:
    whisper_model = None


# =============================
# LOAD ML MODEL (NEW)
# =============================
ml_model = None
try:
    ml_model = joblib.load("model.pkl")
    print("✅ ML model loaded")
except:
    print("⚠️ ML model not found, using fallback")


# =============================
# NORMALIZATION
# =============================
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-6)


# =============================
# SNR ESTIMATION
# =============================
def estimate_snr(y):
    signal_power = np.mean(y ** 2)

    noise = y[np.abs(y) < np.percentile(np.abs(y), 20)]
    noise_power = np.mean(noise ** 2) if len(noise) > 0 else 1e-6

    snr = 10 * np.log10(signal_power / (noise_power + 1e-6))
    return snr


# =============================
# TRANSCRIPTION (WHISPER)
# =============================
def transcribe_audio(file_path):

    if whisper_model is None:
        return "Transcription unavailable", 0.5

    try:
        result = whisper_model.transcribe(
            file_path,
            verbose=False,
            fp16=False
        )

        segments = result.get("segments", [])

        full_text = " ".join([seg["text"].strip() for seg in segments])
        full_text = re.sub(r"\s+", " ", full_text).strip().capitalize()

        probs = []

        for seg in segments:
            if "avg_logprob" in seg:
                prob = np.exp(seg["avg_logprob"])
                probs.append(prob)

        confidence = float(np.mean(probs)) if probs else 0.5
        confidence = np.clip(confidence, 0, 1)

        return full_text, confidence

    except Exception as e:
        return f"Transcription error: {str(e)}", 0.5


# =============================
# AI FEEDBACK
# =============================
def generate_feedback(features, snr, confidence):
    feedback = []

    if snr < 10:
        feedback.append("High background noise detected")

    if features['zcr'] > 0.1:
        feedback.append("Signal contains excessive noise fluctuations")

    if features['rms'] < 0.02:
        feedback.append("Audio volume is too low")

    if confidence < 0.5:
        feedback.append("Speech is not clearly understandable")

    if not feedback:
        feedback.append("Audio quality is good and clear")

    return feedback


# =============================
# QUALITY SCORE (OLD LOGIC KEPT)
# =============================
def compute_quality_score(features, y=None, file_path=None):

    spec_bw = features['spec_bw']
    zcr = features['zcr']
    rms = features['rms']

    clarity = 1 - normalize(spec_bw, 1000, 4000)
    noise = 1 - normalize(zcr, 0.01, 0.2)
    energy = normalize(rms, 0.01, 0.1)

    snr_score = normalize(estimate_snr(y), 0, 30) if y is not None else 0.5

    _, confidence = transcribe_audio(file_path) if file_path else ("", 0.5)

    score = (
        0.25 * clarity +
        0.2 * noise +
        0.15 * energy +
        0.2 * snr_score +
        0.2 * confidence
    )

    return float(np.clip(score * 100, 0, 100))


# =============================
# OLD CLASSIFICATION (FALLBACK)
# =============================
def classify_quality(score):
    if score >= 75:
        return "Good"
    elif score >= 50:
        return "Moderate"
    else:
        return "Poor"


# =============================
# 🔥 ML PREDICTION (NEW)
# =============================
def predict_quality_ml(features):
    """
    Predict quality using trained ML model
    """

    if ml_model is None:
        return "Moderate"  # fallback if model not loaded

    feature_vector = np.concatenate([
        features["mfcc_mean"],
        [
            features["spec_centroid"],
            features["zcr"],
            features["rms"],
            features["spec_bw"]
        ]
    ])

    prediction = ml_model.predict([feature_vector])[0]
    return prediction