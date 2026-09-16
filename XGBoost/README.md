# AI-Based Groundwater Depletion & Borewell Failure Predictor
**B.Tech Capstone Project MVP**

## Setup & Execution on Windows:
1. Double-click `run_windows.bat` (auto-creates venv, installs requirements, trains models, launches Streamlit).
2. Or run manually in CMD:
   ```cmd
   pip install -r requirements.txt
   python src/train_models.py
   streamlit run app.py
   ```

## Features Implemented:
- Data validation and clean preprocessing
- Diurnal temperature feature engineering
- XGBoost Depth Regressor (RMSE, MAE, R²)
- XGBoost Success Classifier (Accuracy, Precision, Recall, F1, ROC-AUC)
- Streamlit interactive dashboard with farmer recommendations
- Saturated zone drilling buffer recommendations
