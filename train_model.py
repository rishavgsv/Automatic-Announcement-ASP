import pandas as pd
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.feature_extraction import extract_features

df = pd.read_csv("data/labels.csv")

X, y = [], []

print("🚀 Extracting features...")

for _, row in df.iterrows():
    try:
        audio, sr = librosa.load(row["file"], sr=16000)
        features = extract_features(audio, sr)

        vector = np.concatenate([
            features["mfcc_mean"],
            [features["spec_centroid"], features["zcr"], features["rms"], features["spec_bw"]]
        ])

        X.append(vector)
        y.append(row["label"])

    except Exception as e:
        print("Error:", e)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 REPORT:\n", classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")

print("✅ Model saved")