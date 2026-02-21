import re
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def clean_log(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{1,3}(\.\d{1,3}){3}', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def load_logs(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        # Always use 'message' column if present
        if "message" in df.columns:
            df = df[["message"]].rename(columns={"message": "raw"})
        else:
            # Fallback to last column
            text_col = df.columns[-1]
            df = df[[text_col]].rename(columns={text_col: "raw"})
    else:
        # TXT file
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f]
        df = pd.DataFrame({"raw": lines})

    return df

def preprocess(df):
    df["clean"] = df["raw"].apply(clean_log)
    df = df[df["clean"] != ""]
    return df.reset_index(drop=True)

def train_model(df):
    df = preprocess(df)
    vect = TfidfVectorizer(max_features=2000, stop_words="english")
    X = vect.fit_transform(df["clean"])
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    joblib.dump(vect, VECT_PATH)
    joblib.dump(model, MODEL_PATH)
    return model, vect

def predict(df):
    df = preprocess(df)
    vect = joblib.load(VECT_PATH)
    model = joblib.load(MODEL_PATH)
    X = vect.transform(df["clean"])
    scores = model.decision_function(X)
    preds = model.predict(X)
    df["anomaly_score"] = scores
    df["anomaly"] = ["Normal" if p == 1 else "Anomaly" for p in preds]
    return df
