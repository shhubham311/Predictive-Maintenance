from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import advanced_feature_engineering

app = FastAPI(title="Predictive Maintenance API")

# Load Model
model_path = "models/advanced_model.pkl"
artifacts = joblib.load(model_path)
model = artifacts['model']
scaler = artifacts['scaler']
training_cols = artifacts['training_cols']

class SensorData(BaseModel):
    setting1: float
    setting2: float
    setting3: float
    s1: float = 518.67
    s2: float = 641.82
    s3: float = 1589.7
    s4: float = 1400.6
    s5: float = 14.62
    s6: float = 21.61
    s7: float = 554.36
    s8: float = 2388.06
    s9: float = 9046.2
    s10: float = 1.3
    s11: float = 47.3
    s12: float = 521.66
    s13: float = 2388.02
    s14: float = 8138.62
    s15: float = 8.3199
    s16: float = 0.03
    s17: float = 391.0
    s18: float = 2388.0
    s19: float = 100.0
    s20: float = 39.0
    s21: float = 23.4194

@app.post("/predict")
def predict_rul(data: SensorData):
    try:
        df_input = pd.DataFrame([data.dict()])
        df_processed = advanced_feature_engineering(df_input, training=False)
        
        # Align columns
        for col in training_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
                
        X = df_processed[training_cols]
        X_scaled = scaler.transform(X)
        
        import xgboost as xgb
        dtest = xgb.DMatrix(X_scaled)
        
        # 1. Anomaly Check
        is_anomaly = False
        if 'anomaly_detector' in artifacts:
            iso = artifacts['anomaly_detector']
            # FIX: Explicitly cast to Python bool to avoid FastAPI serialization error
            is_anomaly = bool(iso.predict(X_scaled)[0] == -1)
            
        # 2. RUL Prediction
        prediction = model.predict(dtest)[0]
        
        # 3. Status
        status = "Normal"
        if is_anomaly:
            status = "Anomaly Detected"
        elif prediction < 50:
            status = "Critical"
            
        return {
            "predicted_rul": float(prediction),
            "anomaly": is_anomaly,
            "status": status
        }
    except Exception as e:
        # Return detailed error for debugging
        raise HTTPException(status_code=500, detail=str(e))