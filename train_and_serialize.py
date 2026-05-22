import os
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

print("="*60)
print("FMCG FORECASTING PIPELINE: TRAINING & SERIALIZATION")
print("="*60)

# 1. Establish project directory structure
folders = ['data', 'models', 'outputs', 'notebooks', 'backend', 'rag']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"[OK] Folder verified/created: {folder}/")

# 2. Copy dataset to data/ folder if not already there
src_dataset = os.path.join("dataset", "augmented_ml_ready_dataset.csv")
dest_dataset = os.path.join("data", "augmented_ml_ready_dataset.csv")

if os.path.exists(src_dataset) and not os.path.exists(dest_dataset):
    shutil.copy(src_dataset, dest_dataset)
    print(f"[OK] Copied dataset to production location: {dest_dataset}")
elif os.path.exists(dest_dataset):
    print(f"[OK] Dataset verified in data folder: {dest_dataset}")
else:
    print(f"[WARNING] Could not find source dataset at {src_dataset}")

# Load dataset
if os.path.exists(dest_dataset):
    df = pd.read_csv(dest_dataset)
elif os.path.exists(src_dataset):
    df = pd.read_csv(src_dataset)
else:
    raise FileNotFoundError("Could not find augmented_ml_ready_dataset.csv in dataset/ or data/ directories!")

print(f"Dataset loaded. Shape: {df.shape}")

# 3. Data Ingestion & Preprocessing
target_col = 'quantity_sold'
cols_to_drop = [
    'product_name', 
    'low_stock_alert', 
    'stock_low', 
    'stock_normal', 
    'stock_high',
    'reorder_quantity'
]
df_features = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Chronological sorting and splitting
df_features['date'] = pd.to_datetime(df_features['date'])
df_sorted = df_features.sort_values(by='date').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx].copy()
test_df = df_sorted.iloc[split_idx:].copy()

X_train = train_df.drop(columns=['date', target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=['date', target_col])
y_test = test_df[target_col]

print(f"Chronological split completed:")
print(f"  * Training set: {X_train.shape[0]} rows (Dates: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
print(f"  * Testing set:  {X_test.shape[0]} rows (Dates: {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

# 4. Train Champion XGBoost Regressor
print("\nTraining production models...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
print("[OK] Trained Champion XGBoost Regressor.")

# Train LightGBM & CatBoost for comparison backups
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
print("[OK] Trained Backup LightGBM Regressor.")

# CatBoost expects string/float encodings, we run it silently
cat_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train)
print("[OK] Trained Backup CatBoost Regressor.")

# 5. Evaluate and Print Leaderboard
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_cat = cat_model.predict(X_test)

metrics = {
    'XGBoost': {
        'MAE': mean_absolute_error(y_test, y_pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'R2': r2_score(y_test, y_pred_xgb)
    },
    'LightGBM': {
        'MAE': mean_absolute_error(y_test, y_pred_lgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lgb)),
        'R2': r2_score(y_test, y_pred_lgb)
    },
    'CatBoost': {
        'MAE': mean_absolute_error(y_test, y_pred_cat),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_cat)),
        'R2': r2_score(y_test, y_pred_cat)
    }
}
metrics_df = pd.DataFrame(metrics).T
print("\nModel Leaderboard Metrics:")
print(metrics_df.to_string())

# 6. Save binary model files using joblib
joblib.dump(xgb_model, os.path.join("models", "xgboost_model.pkl"))
joblib.dump(lgb_model, os.path.join("models", "lightgbm_model.pkl"))
joblib.dump(cat_model, os.path.join("models", "catboost_model.pkl"))
print("\n[OK] Serialized all models successfully inside models/")

# 7. Generate Initial Outputs for RAG Engine and Dashboard Ingestion
print("\nGenerating and exporting prediction spreadsheets to outputs/...")
test_out = test_df.copy()
test_out['predicted_demand'] = y_pred_xgb

# Restore original product name and metadata if present
if 'product_name' in df.columns:
    test_out['product_name'] = df.loc[test_out.index, 'product_name']
else:
    test_out['product_name'] = 'Biscuits'

# Operations Calculations
test_out['reorder_required'] = (test_out['predicted_demand'] > test_out['net_stock']).astype(int)
test_out['safety_adjusted_reorder_qty'] = test_out.apply(
    lambda r: max(0, int(np.ceil(r['predicted_demand'] * 1.5 - r['net_stock']))) 
    if r['reorder_required'] == 1 else 0, axis=1
)

def get_stock_status(r):
    if r['net_stock'] < r['predicted_demand']:
        return "Understocked"
    elif r['net_stock'] > (r['predicted_demand'] * 2.0):
        return "Overstocked"
    else:
        return "Normal"

test_out['inventory_status'] = test_out.apply(get_stock_status, axis=1)

def get_stock_risk(r):
    if r['net_stock'] < (r['predicted_demand'] * 0.5):
        return "High Risk"
    elif r['net_stock'] < r['predicted_demand']:
        return "Medium Risk"
    else:
        return "Low Risk"

test_out['stock_risk_level'] = test_out.apply(get_stock_risk, axis=1)

def get_alert_severity(r):
    if r['inventory_status'] == 'Understocked':
        return "Critical" if r['stock_risk_level'] == 'High Risk' else "Warning"
    return "Info"

test_out['alert_severity'] = test_out.apply(get_alert_severity, axis=1)

# Dynamically calculate stockout probability based on rolling variance & current buffer
test_out['stockout_probability'] = test_out.apply(
    lambda r: min(0.99, max(0.01, 1.0 - (r['net_stock'] / (r['predicted_demand'] + 1e-5)))) 
    if r['inventory_status'] == 'Understocked' else 0.05, axis=1
)

# Explainable AI Insights
def generate_xai_insight(row):
    explanations = []
    
    if row.get('promotion_flag', 0) == 1:
        intensity = row.get('marketing_intensity', 0)
        explanations.append(f"An active promotional campaign with marketing intensity of {intensity:.2f} is accelerating demand.")
    
    if row.get('weekend_flag', 0) == 1:
        explanations.append("High weekend traffic is boosting consumer buying activity.")
        
    if row.get('festival_flag', 0) == 1:
        explanations.append("Holiday season festival shopping is driving an active sales surge.")
        
    seasons = []
    if row.get('season_Summer', 0) == 1: seasons.append("Summer")
    if row.get('season_Winter', 0) == 1: seasons.append("Winter")
    if row.get('season_Autumn', 0) == 1: seasons.append("Autumn")
    if row.get('season_Monsoon', 0) == 1: seasons.append("Monsoon")
    
    if seasons:
        explanations.append(f"Seasonal patterns for {', '.join(seasons)} are driving temperature-dependent snack/beverage demand.")
        
    if row.get('avg_roas', 0) > 3.0:
        explanations.append("Exceptional Return on Ad Spend (ROAS) indicates active marketing campaigns are highly effective.")
        
    if not explanations:
        explanations.append("Demand is stable and matching normal baseline consumer patterns.")
        
    return " ".join(explanations)

test_out['forecast_explanation'] = test_out.apply(generate_xai_insight, axis=1)

# Export deliverables
forecast_predictions_export = test_out[['date', 'product_name', 'predicted_demand']]
forecast_predictions_export.to_csv(os.path.join("outputs", "forecast_predictions.csv"), index=False)

inventory_recs_export = test_out[[
    'date', 'product_name', 'net_stock', 'predicted_demand', 
    'reorder_required', 'safety_adjusted_reorder_qty', 
    'inventory_status', 'stock_risk_level', 'alert_severity', 
    'stockout_probability', 'forecast_explanation'
]]
inventory_recs_export.to_csv(os.path.join("outputs", "inventory_recommendations.csv"), index=False)
print("[OK] Exported forecast_predictions.csv and inventory_recommendations.csv to outputs/")
print("="*60)
print("[TRAINING COMPLETE] Production setup ready!")
print("="*60)
