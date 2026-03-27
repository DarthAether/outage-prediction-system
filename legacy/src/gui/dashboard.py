import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Disaster Early Warning System", layout="centered")

st.title("⚡ Power Outage Early Warning System")
st.markdown("Predict outage risk and trigger disaster response **before** failures occur.")

st.subheader("Storm Severity Input")

magnitude = st.slider("Storm Magnitude", 0.0, 150.0, 50.0)
damage = st.number_input("Estimated Property Damage ($)", min_value=0.0, step=1000.0)

if st.button("Predict Outage Risk"):
    payload = {
        "magnitude": magnitude,
        "damage_property": damage
    }

    response = requests.post(API_URL, json=payload).json()

    risk = response["risk_level"]
    prob = response["outage_risk_probability"]

    if risk == "RED":
        st.error(f"🚨 RED ALERT — Outage Risk: {prob}")
    elif risk == "ORANGE":
        st.warning(f"⚠️ ORANGE ALERT — Outage Risk: {prob}")
    elif risk == "YELLOW":
        st.info(f"🟡 YELLOW ALERT — Outage Risk: {prob}")
    else:
        st.success(f"🟢 GREEN — Outage Risk: {prob}")

    st.subheader("Recommended Actions")
    for action in response["recommended_actions"]:
        st.write(f"- {action}")
