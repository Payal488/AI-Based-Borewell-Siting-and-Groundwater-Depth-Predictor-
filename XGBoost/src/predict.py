import os
import joblib
import pandas as pd
from src.data_preprocessing import FEATURE_COLS

def calculate_recommended_drilling_depth(water_depth, well_depth_hint):
    buffer = 15.0 if water_depth < 10 else (18.0 if water_depth < 20 else 22.0)
    rec = round(water_depth + buffer, 1)
    if well_depth_hint > 0 and rec < well_depth_hint:
        rec = float(well_depth_hint)
    return rec, buffer

def predict_borewell_outcome(input_dict, models_dir="models"):
    reg = joblib.load(os.path.join(models_dir, "depth_regressor.joblib"))
    clf = joblib.load(os.path.join(models_dir, "success_classifier.joblib"))

    t2m_max = float(input_dict.get('t2m_max', 35.0))
    t2m_min = float(input_dict.get('t2m_min', 22.0))
    sample = {
        'rainfall': float(input_dict.get('rainfall', 2.0)),
        't2m': float(input_dict.get('t2m', 28.0)),
        't2m_max': t2m_max,
        't2m_min': t2m_min,
        'month': int(input_dict.get('month', 6)),
        'latitude': float(input_dict.get('latitude', 17.5)),
        'longitude': float(input_dict.get('longitude', 78.0)),
        'wellDepth': float(input_dict.get('wellDepth', 30.0)),
        'wellAquiferType_encoded': int(input_dict.get('wellAquiferType_encoded', 0)),
        'State_encoded': 0, 'District_encoded': 0,
        'temp_range': max(0.0, t2m_max - t2m_min)
    }
    df = pd.DataFrame([sample])[FEATURE_COLS]
    pred_depth = max(0.5, round(float(reg.predict(df)[0]), 2))
    probs = clf.predict_proba(df)[0]
    fail_p = float(probs[0])
    succ_p = float(probs[1]) if len(probs) > 1 else (1.0 - fail_p)

    rec_m, buf = calculate_recommended_drilling_depth(pred_depth, sample['wellDepth'])
    viability = "High" if succ_p >= 0.70 else ("Moderate" if succ_p >= 0.40 else "Low")
    msg = "Favorable water availability" if viability == "High" else ("Moderate table; monitor seasonal recharge" if viability == "Moderate" else "High risk of dry well; deep depletion detected")

    return {
        "success_probability": round(succ_p * 100, 1),
        "failure_probability": round(fail_p * 100, 1),
        "predicted_groundwater_depth_m": pred_depth,
        "recommended_drilling_depth_m": rec_m,
        "viability": viability,
        "viability_message": msg,
        "disclaimer": "AI statistical predictions are advisory only. Consult a certified hydrogeologist."
    }
