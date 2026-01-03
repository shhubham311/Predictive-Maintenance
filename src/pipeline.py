import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def advanced_feature_engineering(df, training=True):
    """
    Adds interaction terms and rolling/lag features.
    """
    df = df.copy()
    
    # 1. Interaction Features (Safe for both Training & Inference)
    df['setting1_x_s1'] = df['setting1'] * df['s1']
    df['setting2_x_s2'] = df['setting2'] * df['s2']
    df['setting3_x_s3'] = df['setting3'] * df['s3']
    
    # 2. Complex Time-Series Features
    if training:
        sensor_cols = [f's{i}' for i in range(1, 22)]
        window_size = 5
        
        for col in sensor_cols:
            # Rolling stats
            df[f'{col}_rolling_mean'] = df.groupby('engine_id')[col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean())
            
            # Lag features
            df[f'{col}_lag1'] = df.groupby('engine_id')[col].shift(1)
        
        # Fill NaNs created by lags
        df = df.fillna(0)
    else:
        # INFERENCE FIX:
        # We don't have history for a single row. 
        # Instead of filling with 0 (which confuses model), 
        # we assume "Steady State" -> Lag = Current, Rolling = Current.
        sensor_cols = [f's{i}' for i in range(1, 22)]
        
        for col in sensor_cols:
            df[f'{col}_rolling_mean'] = df[col] # Assume average is current
            df[f'{col}_lag1'] = df[col]         # Assume previous was same as current
    
    return df

class PredictiveMaintenancePipeline:
    def __init__(self, model_path):
        """
        Load the pre-trained pipeline artifacts
        """
        self.artifacts = joblib.load(model_path)
        self.model = self.artifacts['model']
        self.feature_cols = self.artifacts['feature_cols']
        self.training_cols = self.artifacts['training_cols'] 
        
        # Load Scaler
        self.scaler = self.artifacts['scaler']
        
        # Load Anomaly Detector (Hybrid System)
        self.iso_forest = self.artifacts.get('anomaly_detector', None)
        
    def preprocess(self, input_data):
        """
        Prepare input data. Handles single-row inputs from the web app.
        """
        df = pd.DataFrame([input_data])
        
        # Apply Feature Engineering (Inference Mode)
        df = advanced_feature_engineering(df, training=False)
        
        # Align columns with training data
        # The training data had 'rolling_mean' and 'lag' columns. 
        # The single-row input does not. We must create those columns and fill with 0.
        for col in self.training_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the columns actually used by the model
        X = df[self.training_cols]
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict(self, input_data):
        """
        Run Prediction (Hybrid: RUL + Anomaly)
        """
        X_scaled = self.preprocess(input_data)
        
        # 1. Anomaly Detection (Unsupervised)
        is_anomaly = False
        anomaly_score = 0
        if self.iso_forest is not None:
            # -1 is anomaly, 1 is normal
            pred = self.iso_forest.predict(X_scaled)[0]
            if pred == -1:
                is_anomaly = True
            # Get anomaly score (lower is more anomalous)
            anomaly_score = self.iso_forest.score_samples(X_scaled)[0]

        # 2. RUL Prediction (Supervised)
        import xgboost as xgb
        dtest = xgb.DMatrix(X_scaled)
        prediction = self.model.predict(dtest)[0]
            
        # 3. Hybrid Logic for Status
        if is_anomaly:
            status = "🚨 SENSOR ANOMALY DETECTED (UNSAFE)"
            status_code = 2 # High Alert
        elif prediction < 50:
            status = "⚠️ CRITICAL RUL (MAINTENANCE NEEDED)"
            status_code = 1 # Warning
        else:
            status = "✅ NORMAL OPERATION"
            status_code = 0 # Normal
            
        return float(prediction), status, status_code, float(anomaly_score)