from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from src.api.alerting import classify_risk, response_actions

MODEL_PATH = "models/random_forest.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Early Warning Power Outage Prediction System",
    description="Predicts power outage risk and generates disaster-response alerts",
    version="2.0"
)

class StormEvent(BaseModel):
    magnitude: float
    damage_property: float

class AlertResponse(BaseModel):
    outage_risk_probability: float
    risk_level: str
    recommended_actions: list[str]

@app.post("/predict", response_model=AlertResponse)
def predict(event: StormEvent):
    X = [[event.magnitude, event.damage_property]]
    prob = model.predict_proba(X)[0][1]

    risk = classify_risk(prob)
    actions = response_actions(risk)

    return {
        "outage_risk_probability": round(prob, 3),
        "risk_level": risk,
        "recommended_actions": actions
    }
