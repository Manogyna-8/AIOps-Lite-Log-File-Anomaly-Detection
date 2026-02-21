
from flask import Flask, render_template, request
import os
from core.ml_engine import load_logs, train_model, predict

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    df = load_logs(file_path)
    train_model(df)

    return "Model trained successfully!"

@app.route("/predict", methods=["POST"])
def detect():
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    df = load_logs(file_path)
    result = predict(df)

    return render_template("results.html", tables=result.to_html(classes="table"))

if __name__ == "__main__":
    app.run(debug=True)
