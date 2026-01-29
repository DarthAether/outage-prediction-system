from fastapi import FastAPI
from pydantic import BaseModel
import joblib

MODEL_PATH = "models/random_forest.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Outage Risk Prediction API",
    description="Predicts outage risk from storm severity indicators",
    version="1.0"
)

class StormEvent(BaseModel):
    magnitude: float
    damage_property: float

class Prediction(BaseModel):
    outage_risk_probability: float
    outage_risk_class: int

@app.post("/predict", response_model=Prediction)
def predict(event: StormEvent):
    X = [[event.magnitude, event.damage_property]]
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    return {
        "outage_risk_probability": prob,
        "outage_risk_class": pred
    }
