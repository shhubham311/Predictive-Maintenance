import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_feature_importance(model, feature_cols, save_path):
    """Generates and saves a feature importance bar chart."""
    
    # Handle XGBoost Booster vs Scikit-Learn style
    if hasattr(model, 'get_score'):
        # XGBoost Booster
        scores = model.get_score(importance_type='gain') # 'gain' is better for resume
        # Map scores back to feature names
        importances = [scores.get(f'f{i}', 0) for i in range(len(feature_cols))]
    else:
        # Scikit-Learn
        importances = model.feature_importances_

    # Get Top 15 indices
    indices = (np.array(importances)).argsort()[-15:]

    plt.figure(figsize=(10, 6))
    plt.title('Top 15 Feature Importances (Gain)')
    plt.barh(range(len(indices)), [importances[i] for i in indices], align='center')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true, y_pred, save_path):
    """Generates Actual vs Predicted scatter and Residual Histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.3)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_title('Actual vs Predicted RUL')
    axes[0].set_xlabel('Actual RUL')
    axes[0].set_ylabel('Predicted RUL')
    
    # 2. Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Prediction Error')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()