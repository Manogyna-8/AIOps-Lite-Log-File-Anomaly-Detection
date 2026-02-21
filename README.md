# 🔍 AIOps Lite – Log File Anomaly Detection using Machine Learning

## 📌 Overview

AIOps Lite is a Machine Learning-based web application that detects anomalies in IT log files using the Isolation Forest algorithm.

The system preprocesses unstructured logs using NLP techniques and automatically identifies suspicious patterns to assist in proactive system monitoring.

---

## 🚀 Features

- Upload log files (.csv / .txt)
- Text preprocessing (regex cleaning, normalization)
- TF-IDF vectorization for feature extraction
- Isolation Forest–based anomaly detection
- 5% contamination threshold configuration
- Real-time anomaly detection via Flask interface
- Anomaly score generation
- Human-readable classification (Normal / Anomaly)

---

## 🏗 System Architecture

**Pipeline Flow:**

Upload → Preprocess → Vectorize → Train → Predict → Display

**Core Components:**

- Log Loader  
- Log Preprocessor  
- Isolation Forest Model  
- Prediction Engine  
- Flask Web Interface  

---

## 📊 Model Performance

- Processed up to 4,000 logs in under 2 seconds  
- Designed to handle up to 50,000+ log entries per session  
- Detection response time: 3–5 seconds (moderate datasets)  
- Contamination threshold: 5%  
- Accuracy: 89.7%  
- Weighted F1-score: 0.87  

---

## 🛠 Tech Stack

**Backend:** Python, Flask  
**Machine Learning:** Scikit-learn (Isolation Forest)  
**NLP:** TF-IDF Vectorization, Regex  
**Data Handling:** Pandas, NumPy  
**Model Persistence:** Joblib  

---

## 📁 Project Structure

```
AIOps_Lite/
│
├── core/
│   └── ml_engine.py
│
├── data/
│   └── aiops_logs.csv
│
├── model/
├── templates/
│   ├── index.html
│   └── results.html
│
├── uploads/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 How to Run Locally

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
python app.py
```

3. Open in browser:

```
http://127.0.0.1:5000
```

---

## 🔮 Future Improvements

- Real-time streaming log analysis  
- Deep learning-based anomaly detection  
- Dashboard visualization  
- Cloud deployment  
- User authentication system  

---

## 👩‍💻 Author

Manogyna A  
