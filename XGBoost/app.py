import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.predict import predict_borewell_outcome
from src.train_models import train_and_save_models

st.set_page_config(page_title="Borewell Success Predictor", page_icon="💧", layout="wide")

st.title("💧 Borewell Success Predictor")
st.caption("AI-Assisted Groundwater Depletion & Drilling Feasibility Advisor")

models_dir = "models"
if not os.path.exists(os.path.join(models_dir, "depth_regressor.joblib")):
    with st.spinner("Training models for first time..."):
        train_and_save_models("data/groundwater_data.csv", models_dir)

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        latitude = st.number_input("Latitude (°N)", 8.0, 35.0, 17.38, 0.01)
        longitude = st.number_input("Longitude (°E)", 68.0, 97.0, 78.43, 0.01)
        well_depth = st.number_input("Planned Well Depth (m)", 5.0, 250.0, 30.0, 1.0)
    with c2:
        month = st.selectbox("Month", list(range(1, 13)), index=4)
        rainfall = st.number_input("24h Rainfall (mm)", 0.0, 200.0, 1.5, 0.5)
        t2m = st.number_input("Mean Temp (°C)", 10.0, 50.0, 28.5, 0.5)
    with c3:
        t2m_max = st.number_input("Max Temp (°C)", 15.0, 55.0, 36.0, 0.5)
        t2m_min = st.number_input("Min Temp (°C)", 5.0, 40.0, 22.0, 0.5)
        aquifer = st.selectbox("Aquifer Type", [0, 1], format_func=lambda x: "Hard-rock (0)" if x == 0 else "Alluvial (1)")

    submitted = st.form_submit_button("Predict Borewell Outcome", use_container_width=True)

if submitted:
    res = predict_borewell_outcome({
        'rainfall': rainfall, 't2m': t2m, 't2m_max': t2m_max, 't2m_min': t2m_min,
        'month': month, 'latitude': latitude, 'longitude': longitude,
        'wellDepth': well_depth, 'wellAquiferType_encoded': aquifer,
        'State_encoded': 0, 'District_encoded': 0
    }, models_dir)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Success Probability", f"{res['success_probability']}%")
    c2.metric("Failure Risk", f"{res['failure_probability']}%")
    c3.metric("Est. Water Depth", f"{res['predicted_groundwater_depth_m']} m")
    c4.metric("Recommended Drilling", f"{res['recommended_drilling_depth_m']} m")

    st.write(f"**Viability:** {res['viability']} — {res['viability_message']}")
    st.info(res['disclaimer'])
