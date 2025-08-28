from fastapi import FastAPI
import pandas as pd
import torch
from src.model import load_model

app = FastAPI()
model = load_model("models/fraud_model.pkl")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    x = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).numpy()[0][0]
    return {"fraud_probability": float(pred)}
