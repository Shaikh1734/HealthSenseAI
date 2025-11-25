# HealthSenseAI: Symptom Classification Agent (Google Cloud Run + ADK)

This project is an ML-powered agent that predicts probable diseases from patient symptom descriptions using a trained machine learning model. It is designed for *production-ready deployment on Google Cloud Run* with a built-in chat interface, using the [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/).

---

## Features

- *Free-text Symptom Classification:* Accepts user-entered symptom descriptions and returns a disease prediction.
- *Interactive Web UI:* User-friendly chat interface for rapid health query testing.
- *Cloud Native:* Containerized and served via Google Cloud Run.
- *Fast, reproducible ML pipeline:* Uses scikit-learn, joblib for rapid prototyping and fast inference.

---

## Project Structure
HealthSenseAI/
├── agent.py                 # ADK agent logic & ML integration
├── model.pkl                # Trained ML model (joblib format)
├── vectorizer.pkl           # Trained text vectorizer (joblib format)
├── requirements.txt         # Python dependencies
├── Dockerfile               # (Optional) Custom Docker image config
├── main.ipynb               # (Optional) ML training & demo notebook
├── synthetic_symptom_data_10000.xls  # (Optional) Training data sample
└── README.md                # This doc
