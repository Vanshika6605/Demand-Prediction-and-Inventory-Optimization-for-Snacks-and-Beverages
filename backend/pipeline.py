import os
import joblib
import pandas as pd
import numpy as np

# Feature Column Alignment exactly as used during training
FEATURE_COLUMNS = [
    'day_of_week', 'month', 'quarter', 'week_of_year', 'weekend_flag',
    'festival_flag', 'lag_1', 'lag_7', 'lag_14', 'rolling_7_mean',
    'rolling_14_mean', 'rolling_7_std', 'promotion_flag', 'campaign_active',
    'marketing_intensity', 'avg_roas', 'net_stock', 'stock_turnover_rate',
    'avg_unit_price', 'season_Autumn', 'season_Monsoon', 'season_Summer',
    'season_Winter', 'category_Cold Drinks & Juices',
    'category_Hot Beverages', 'category_Snacks & Munchies'
]

# Ensure the feature matrix excludes target-leaking fields
MODEL_FEATURES = [col for col in FEATURE_COLUMNS if col not in ['net_stock']]

def predict_demand(features_dict: dict, model_path: str = "models/xgboost_model.pkl") -> float:
    \"\"\"
    Loads serialized model and performs inference for demand forecasting.
    \"\"\"
    if not os.path.exists(model_path):
        # Graceful fallback in case champion file is missing
        backup_paths = [
            "models/lightgbm_model.pkl",
            "models/catboost_model.pkl"
        ]
        for path in backup_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            raise FileNotFoundError(f"No trained model found at {model_path} or backups!")
            
    model = joblib.load(model_path)
    
    # Align incoming dictionary into features dataframe
    df_in = pd.DataFrame([features_dict])
    
    # Standardize columns and fill missing with default zeros
    for col in MODEL_FEATURES:
        if col not in df_in.columns:
            df_in[col] = 0.0
            
    X = df_in[MODEL_FEATURES]
    pred = float(model.predict(X)[0])
    return max(0.0, pred)

def generate_inventory_alerts(predicted_demand: float, net_stock: float) -> dict:
    \"\"\"
    Evaluates net stock against predicted demand to trigger alerts and estimate stockout risk.
    \"\"\"
    # Classify Inventory Status
    if net_stock < predicted_demand:
        status = "Understocked"
    elif net_stock > (predicted_demand * 2.0):
        status = "Overstocked"
    else:
        status = "Normal"
        
    # Classify Stock Risk Level
    if net_stock < (predicted_demand * 0.5):
        risk_level = "High Risk"
    elif net_stock < predicted_demand:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
        
    # Determine Alert Severity
    if status == "Understocked":
        severity = "Critical" if risk_level == "High Risk" else "Warning"
    else:
        severity = "Info"
        
    # Dynamically Estimate Stockout Probability
    if status == "Understocked":
        stockout_prob = min(0.99, max(0.01, 1.0 - (net_stock / (predicted_demand + 1e-5))))
    else:
        stockout_prob = 0.05
        
    # Define Color Codes for UI Dashboard Rendering
    color_map = {
        "High Risk": "#ef4444",   # Tailwind Red-500
        "Medium Risk": "#f97316", # Tailwind Orange-500
        "Low Risk": "#10b981"     # Tailwind Emerald-500
    }
    
    severity_color_map = {
        "Critical": "#ef4444",
        "Warning": "#f97316",
        "Info": "#3b82f6"         # Tailwind Blue-500
    }
    
    return {
        "inventory_status": status,
        "stock_risk_level": risk_level,
        "alert_severity": severity,
        "stockout_probability": float(stockout_prob),
        "risk_color": color_map[risk_level],
        "severity_color": severity_color_map[severity]
    }

def generate_reorder_recommendations(predicted_demand: float, net_stock: float) -> dict:
    \"\"\"
    Triggers reorder recommendations and applies 1.5x safety stock formula.
    \"\"\"
    reorder_required = int(predicted_demand > net_stock)
    
    if reorder_required == 1:
        # 1.5x safety stock buffer
        qty = max(0, int(np.ceil(predicted_demand * 1.5 - net_stock)))
    else:
        qty = 0
        
    return {
        "reorder_required": bool(reorder_required),
        "safety_adjusted_reorder_qty": qty
    }
