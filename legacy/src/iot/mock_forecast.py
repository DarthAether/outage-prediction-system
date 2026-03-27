import requests
import time
import random

API_URL = "http://127.0.0.1:8000/predict"

def generate_forecast_event():
    return {
        "magnitude": random.uniform(30, 100),
        "damage_property": random.uniform(0, 5_000_000)
    }

while True:
    event = generate_forecast_event()
    response = requests.post(API_URL, json=event).json()

    print("\n--- EARLY WARNING ALERT ---")
    print("Forecasted Storm Severity:", event)
    print("Risk Level:", response["risk_level"])
    print("Recommended Actions:")
    for action in response["recommended_actions"]:
        print("-", action)

    time.sleep(5)
