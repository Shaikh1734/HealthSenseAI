

from fastapi import FastAPI, Request
import numpy as np
import pandas as pd
import joblib

# Load the vectorizer and the model
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "HealthSense FastAPI ML is live!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    # Input should be {"symptom": "some symptom description"}
    symptom = data.get("symptom")
    if not symptom:
        return {"error": "Input JSON must include the 'symptom' field."}
    features = vectorizer.transform([symptom])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
    