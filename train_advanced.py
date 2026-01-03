import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
import optuna
import os
import sys
import matplotlib
matplotlib.use('Agg') # Prevent GUI issues on Mac

# Imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import advanced_feature_engineering
from utils import plot_feature_importance, plot_residuals

def main():
    # 1. Setup Paths
    DATA_DIR = 'data/raw/'
    MODEL_DIR = 'models'
    IMG_DIR = 'img'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    
    MODEL_FILE = os.path.join(MODEL_DIR, 'advanced_model.pkl')
    METRICS_FILE = os.path.join(MODEL_DIR, 'metrics.json')
    
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

    # 2. Load Data
    print("📂 Loading Data...")
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_FD001.txt'), sep=' ', names=columns, index_col=False)
    except FileNotFoundError:
        print("Data not found. Ensure FD001 files are in data/raw/")
        return

    # 3. Calculate RUL
    print("🧮 Calculating RUL...")
    max_cycle = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max']
    train_df = train_df.merge(max_cycle, on='engine_id', how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df = train_df.drop('max', axis=1)

    # 4. Advanced Feature Engineering
    print("⚙️ Engineering Features...")
    train_df = advanced_feature_engineering(train_df, training=True)
    
    feature_cols = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL']]
    X = train_df[feature_cols]
    y = train_df['RUL']

    # 5. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Scale Data
    print("📏 Scaling Data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 7. HYBRID SYSTEM: Train Anomaly Detector
    print("👻 Training Isolation Forest (Anomaly Detection)...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train_scaled)

    # 8. OPTIMIZATION: Optuna Hyperparameter Tuning
    print("🔍 Running Optuna Optimization...")
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': trial.suggest_float('eta', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'seed': 42
        }
        
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        # Note: Removed XGBoostPruningCallback to fix ModuleNotFoundError on Mac M3/Python 3.12
        model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'validation')], verbose_eval=False)
        
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print(f"✅ Best Trial RMSE: {study.best_value:.2f}")
    print(f"📝 Best Params: {study.best_params}")

    # 9. Train Final XGBoost Model with Best Params
    print("🚀 Training Final Model with Best Params...")
    best_params = study.best_params
    best_params['eval_metric'] = 'rmse'
    
    dtrain_final = xgb.DMatrix(X_train_scaled, label=y_train)
    dval_final = xgb.DMatrix(X_val_scaled, label=y_val)
    
    final_model = xgb.train(best_params, dtrain_final, num_boost_round=500, evals=[(dtrain_final, 'train'), (dval_final, 'val')], verbose_eval=False)

    # 10. Evaluate & Generate Metrics
    print("📊 Evaluating Model...")
    val_pred = final_model.predict(dval_final)
    
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    
    metrics_dict = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2_Score': float(r2),
        'Best_Params': study.best_params
    }
    
    print(f"Final RMSE: {rmse:.2f}, R2: {r2:.4f}")

    # 11. Generate Visualizations
    print("🎨 Generating Plots...")
    plot_feature_importance(final_model, feature_cols, os.path.join(IMG_DIR, 'feature_importance.png'))
    plot_residuals(y_val, val_pred, os.path.join(IMG_DIR, 'residual_plot.png'))

    # 12. Save Everything
    print("💾 Saving Artifacts...")
    artifacts = {
        'model': final_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'training_cols': feature_cols,
        'anomaly_detector': iso_forest # Save hybrid component
    }
    
    joblib.dump(artifacts, MODEL_FILE)
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print("✨ Pipeline Complete. Model, Metrics, and Graphs saved.")

if __name__ == "__main__":
    main()