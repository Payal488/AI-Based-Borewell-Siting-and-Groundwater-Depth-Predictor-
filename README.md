AI-Based Groundwater Depletion & Borewell Failure Predictor 

An AI/ML-based decision-support system for estimating groundwater depth and predicting borewell viability using environmental, geographical, temporal, and well-related features.

The project explores two machine learning approaches:

- XGBoost
- Random Forest

An interactive dashboard is also included to demonstrate the prediction workflow.


Project Overview

Borewell drilling decisions are often made using limited information such as local experience or guesswork. This project aims to provide a data-driven approach by combining groundwater, rainfall, temperature, geographical, and well-related information.

The system is designed to:

- Predict estimated groundwater depth.
- Estimate borewell success probability.
- Analyze environmental and geographical factors affecting groundwater availability.
- Compare XGBoost and Random Forest machine learning models.
- Provide predictions through an interactive dashboard.



Objectives

1. Predict groundwater depth at a selected location.
2. Estimate borewell success probability.
3. Use environmental, geographical, temporal, and well-related features.
4. Compare XGBoost and Random Forest models.
5. Present prediction results through an interactive dashboard.
6. Provide a foundation for future groundwater prediction and spatial analysis.



 Machine Learning Models

 1. XGBoost

XGBoost is used for:

- Groundwater-depth regression
- Borewell-success classification

The XGBoost implementation uses:

- XGBRegressor
- XGBClassifier
- 100 estimators
- Random seed = 42

 2. Random Forest

Random Forest was developed as a comparative machine learning approach for:

- Groundwater-depth regression
- Borewell-success classification

The Random Forest implementation is provided as a Jupyter Notebook.

---

Input Features

The models use the following features:

| Feature | Description |
|---|---|
| rainfall | Daily precipitation |
| t2m | Mean 2 m air temperature |
| t2m_max | Maximum daily temperature |
| t2m_min | Minimum daily temperature |
| month | Month/seasonality indicator |
| latitude | Geographic latitude |
| longitude | Geographic longitude |
| wellDepth | Planned/existing well depth |
| wellAquiferType_encoded | Encoded aquifer type |
| State_encoded | Encoded state/administrative zone |
| District_encoded | Encoded district |
| temp_range | Difference between maximum and minimum temperature |



 Project Workflow

```text
Environmental & Geographical Data
              ↓
       Data Preprocessing
              ↓
      Feature Engineering
              ↓
     ┌────────┴─────────┐
     ↓                  ↓
  XGBoost          Random Forest
     ↓                  ↓
     └────────┬─────────┘
              ↓
      Model Predictions
              ↓
 ┌────────────┴─────────────┐
 ↓                          ↓
Groundwater Depth      Borewell Success
Prediction             Probability
              ↓
       Interactive Dashboard
