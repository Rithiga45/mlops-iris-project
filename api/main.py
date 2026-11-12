
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Iris Flower Prediction API",
    description="ML model deployment using FastAPI",
    version="1.0"
)

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(input_data)

    flower_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return {
        "prediction": int(prediction[0]),
        "flower_name": flower_names[int(prediction[0])]
    }
