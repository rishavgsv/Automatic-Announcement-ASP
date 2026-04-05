import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import soundfile as sf
import plotly.graph_objects as go

# -----------------------------
# IMPORT YOUR MODULES
# -----------------------------
from src.preprocessing import load_audio, bandpass_filter, spectral_noise_reduction
from src.feature_extraction import extract_features
from src.model import (
    compute_quality_score,
    estimate_snr,
    transcribe_audio,
    generate_feedback,
    predict_quality_ml   # ✅ NEW ML FUNCTION
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Announcement Analyzer", layout="wide", page_icon="🚆")

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
.metric-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.big-font {
    font-size: 40px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🚆 Automatic Announcement Quality Analyzer")
st.markdown("### AI-powered evaluation of announcement clarity")

st.divider()

# -----------------------------
# GAUGE FUNCTION
# -----------------------------
def show_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Quality Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "📂 Upload one or more audio files",
    type=["wav", "mp3"],
    accept_multiple_files=True
)

results = []

# -----------------------------
# PROCESS FILES
# -----------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # -----------------------------
        # LOAD + PREPROCESS
        # -----------------------------
        y, sr = load_audio(temp_path)

        y_clean = spectral_noise_reduction(
            bandpass_filter(y, sr),
            sr
        )

        # -----------------------------
        # FEATURE EXTRACTION
        # -----------------------------
        features = extract_features(y_clean, sr)

        # -----------------------------
        # ML PREDICTION (NEW)
        # -----------------------------
        label = predict_quality_ml(features)

        # -----------------------------
        # SCORE (KEEP EXISTING LOGIC)
        # -----------------------------
        score = compute_quality_score(features, y_clean, temp_path)

        # -----------------------------
        # ADVANCED METRICS
        # -----------------------------
        snr_value = estimate_snr(y_clean)

        # -----------------------------
        # AI (WHISPER)
        # -----------------------------
        transcript, confidence = transcribe_audio(temp_path)

        # -----------------------------
        # FEEDBACK
        # -----------------------------
        feedback = generate_feedback(features, snr_value, confidence)

        results.append({
            "name": uploaded_file.name,
            "score": score,
            "label": label,
            "audio": y,
            "clean": y_clean,
            "sr": sr,
            "snr": snr_value,
            "transcript": transcript,
            "confidence": confidence,
            "feedback": feedback
        })

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    for i, res in enumerate(results):

        st.divider()
        st.subheader(f"📁 {res['name']}")

        col1, col2 = st.columns(2)

        # -----------------------------
        # ORIGINAL AUDIO
        # -----------------------------
        with col1:
            st.markdown("### 🎧 Original Audio")
            st.audio(uploaded_files[i])

            fig1, ax1 = plt.subplots()
            librosa.display.waveshow(res["audio"], sr=res["sr"], ax=ax1)
            ax1.set_title("Waveform")
            st.pyplot(fig1)

        # -----------------------------
        # CLEANED AUDIO
        # -----------------------------
        with col2:
            st.markdown("### 🧹 Cleaned Audio")

            fig2, ax2 = plt.subplots()
            librosa.display.waveshow(res["clean"], sr=res["sr"], ax=ax2)
            ax2.set_title("Cleaned Waveform")
            st.pyplot(fig2)

        # -----------------------------
        # QUALITY SCORE
        # -----------------------------
        st.markdown("### 📊 Quality Analysis")

        show_gauge(res["score"])

        c1, c2 = st.columns(2)

        c1.markdown(
            f"<div class='metric-box'><div class='big-font'>{res['score']:.2f}</div>Score</div>",
            unsafe_allow_html=True
        )

        c2.markdown(
            f"<div class='metric-box'><div class='big-font'>{res['label']}</div>ML Prediction</div>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # ADVANCED METRICS
        # -----------------------------
        st.markdown("### 🧠 Advanced Evaluation")

        c3, c4 = st.columns(2)
        c3.metric("SNR (dB)", f"{res['snr']:.2f}")
        c4.metric("AI Confidence", f"{res['confidence']:.2f}")

        # -----------------------------
        # AI INSIGHTS
        # -----------------------------
        st.markdown("### 🤖 AI Insights")

        st.markdown("#### 📝 Transcription")
        st.write(res["transcript"])

        st.markdown("#### 💡 Feedback")
        for f in res["feedback"]:
            st.write(f"• {f}")

        # -----------------------------
        # ALERT
        # -----------------------------
        if res["label"] == "Good":
            st.success("✅ High Quality Announcement")
        elif res["label"] == "Moderate":
            st.warning("⚠️ Moderate Quality Announcement")
        else:
            st.error("❌ Poor Quality Announcement")

        # -----------------------------
        # DOWNLOAD CLEANED AUDIO
        # -----------------------------
        temp_out = f"cleaned_{res['name']}.wav"
        sf.write(temp_out, res["clean"], res["sr"])

        with open(temp_out, "rb") as f:
            st.download_button(
                label="⬇️ Download Cleaned Audio",
                data=f,
                file_name=temp_out,
                mime="audio/wav"
            )

    # -----------------------------
    # COMPARISON GRAPH
    # -----------------------------
    if len(results) > 1:
        st.divider()
        st.subheader("📊 Comparison Across Files")

        names = [r["name"] for r in results]
        scores = [r["score"] for r in results]

        fig, ax = plt.subplots()
        ax.bar(names, scores)
        ax.set_title("Quality Score Comparison")
        ax.set_ylabel("Score")
        ax.set_xticklabels(names, rotation=45)

        st.pyplot(fig)