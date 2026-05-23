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
    """
    Loads serialized model and performs inference for demand forecasting.
    """
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
    """
    Evaluates net stock against predicted demand to trigger alerts and estimate stockout risk.
    """
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
    """
    Triggers reorder recommendations and applies 1.5x safety stock formula.
    """
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

def get_product_id(conn, product_name, category_name="Snacks & Munchies", price=15.0):
    """Defensive helper to get or insert product catalog elements."""
    row = conn.execute("SELECT product_id FROM products WHERE product_name = ?", (product_name,)).fetchone()
    if row:
        return row["product_id"]
        
    # get or create category
    cat_row = conn.execute("SELECT category_id FROM categories WHERE category_name = ?", (category_name,)).fetchone()
    if cat_row:
        cat_id = cat_row["category_id"]
    else:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO categories (category_name, is_perishable) VALUES (?, 0)", (category_name,))
        conn.commit()
        cat_id = cursor.lastrowid
        
    # insert product
    cursor = conn.cursor()
    cursor.execute("INSERT INTO products (product_name, category_id, unit_price) VALUES (?,?,?)", (product_name, cat_id, price))
    conn.commit()
    return cursor.lastrowid

def run_pipeline(df: pd.DataFrame, model_path: str = "models/xgboost_model.pkl", 
                 model_id: int = None, user_id: int = None) -> pd.DataFrame:
    """
    Runs batch inference and inventory calculations on a processed DataFrame,
    and inserts the predictions, recommendations, and alerts into fmcg_platform.db.
    """
    df = df.copy()
    
    if not os.path.exists(model_path):
        # Fallback to backups
        backup_paths = ["models/lightgbm_model.pkl", "models/catboost_model.pkl"]
        for path in backup_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            raise FileNotFoundError(f"No trained model found at {model_path} or backups!")
            
    model = joblib.load(model_path)
    X = df[MODEL_FEATURES]
    preds = model.predict(X)
    df['predicted_demand'] = np.clip(preds, 0.0, None)
    
    # Reorders and stock alerts
    df['reorder_required'] = (df['predicted_demand'] > df['net_stock']).astype(int)
    df['safety_adjusted_reorder_qty'] = df.apply(
        lambda r: max(0, int(np.ceil(r['predicted_demand'] * 1.5 - r['net_stock'])))
        if r['reorder_required'] == 1 else 0, axis=1
    )
    
    # Perishable spoilage check
    df['spoilage_risk'] = (
        (df['net_stock'] > df['predicted_demand'] * 1.8) &
        (df['category'].isin(['Cold Drinks & Juices', 'Snacks & Munchies']))
    ).astype(int)
    
    # Abnormal demand check
    df['abnormal_demand'] = (
        (df['predicted_demand'] - df['rolling_14_mean']).abs() > (2 * df['rolling_7_std'])
    ).astype(int)
    
    # Standard classification
    def get_stock_status(r):
        if r['net_stock'] < r['predicted_demand']:
            return "Understocked"
        elif r['net_stock'] > (r['predicted_demand'] * 2.0):
            return "Overstocked"
        else:
            return "Normal"
            
    df['inventory_status'] = df.apply(get_stock_status, axis=1)
    
    def get_stock_risk(r):
        if r['net_stock'] < (r['predicted_demand'] * 0.5):
            return "High Risk"
        elif r['net_stock'] < r['predicted_demand']:
            return "Medium Risk"
        else:
            return "Low Risk"
            
    df['stock_risk_level'] = df.apply(get_stock_risk, axis=1)
    
    # Dynamic alerts construction
    alert_types = []
    severities = []
    alert_messages = []
    recommended_actions = []
    
    for idx, row in df.iterrows():
        reorder_qty = row['safety_adjusted_reorder_qty']
        
        # 1. Critical Stockout (less than 50% demand covered)
        if row['net_stock'] < row['predicted_demand'] * 0.5:
            alert_types.append("Stockout")
            severities.append("Critical")
            alert_messages.append("Critical Stockout Risk: net stock covers less than 50% of forecasted demand.")
            recommended_actions.append(f"Place urgent reorder of {reorder_qty} units immediately.")
        
        # 2. Understocked (less than 100% demand covered)
        elif row['net_stock'] < row['predicted_demand']:
            alert_types.append("Understocked")
            severities.append("Warning")
            alert_messages.append("Understocked: net stock is below forecasted demand.")
            recommended_actions.append(f"Reorder safety buffer of {reorder_qty} units.")
            
        # 3. Spoilage Risk (perishable items significantly overstocked)
        elif row['spoilage_risk'] == 1:
            alert_types.append("Spoilage Risk")
            severities.append("Warning")
            alert_messages.append("High Spoilage Risk: Excess stock on hand for perishable snacking or drink products.")
            recommended_actions.append("Reduce next supplier delivery or run clearance promotion.")
            
        # 4. Abnormal Demand (deviates by more than 2 std dev)
        elif row['abnormal_demand'] == 1:
            alert_types.append("Abnormal Demand")
            severities.append("Warning")
            alert_messages.append("Abnormal Demand: Predicted volume deviates significantly from historical rolling trend.")
            recommended_actions.append("Verify promotional triggers and verify marketing scheduler.")
            
        # 5. General Overstocked (non-perishables or moderate overstock)
        elif row['net_stock'] > row['predicted_demand'] * 2.0:
            alert_types.append("Overstock")
            severities.append("Info")
            alert_messages.append("Overstocked: Net stock levels are elevated compared to predicted demand.")
            recommended_actions.append("Hold back replenishment orders.")
            
        # 6. Healthy Normal
        else:
            alert_types.append("Normal")
            severities.append("Info")
            alert_messages.append("Normal: Inventory levels are healthy and matching forecasted sales.")
            recommended_actions.append("No immediate action required.")
            
    df['alert_type'] = alert_types
    df['alert_severity'] = severities
    df['alert_message'] = alert_messages
    df['recommended_action'] = recommended_actions
    
    df['stockout_probability'] = df.apply(
        lambda r: min(0.99, max(0.01, 1.0 - (r['net_stock'] / (r['predicted_demand'] + 1e-5))))
        if r['inventory_status'] == 'Understocked' else 0.05, axis=1
    )
    
    # UI Color helpers
    color_map = {
        "High Risk": "#ef4444",
        "Medium Risk": "#f97316",
        "Low Risk": "#10b981"
    }
    severity_color_map = {
        "Critical": "#ef4444",
        "Warning": "#f97316",
        "Info": "#3b82f6"
    }
    df['risk_color'] = df['stock_risk_level'].map(color_map)
    df['severity_color'] = df['alert_severity'].map(severity_color_map)
    
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
        
    df['forecast_explanation'] = df.apply(generate_xai_insight, axis=1)
    
    # Write outputs directly to SQL Tables in the SQLite DB
    from database_setup import get_connection, get_active_champion, save_forecasts, save_inventory_recommendations, save_alert
    
    conn = get_connection()
    try:
        # Check / Get model ID
        if model_id is None:
            champ = get_active_champion(conn)
            if champ:
                model_id = champ['model_id']
            else:
                model_id = 1
                
        # 1. Insert forecasts
        forecast_records = []
        for idx, row in df.iterrows():
            p_name = row['product_name']
            c_name = row.get('category', 'Snacks & Munchies')
            price_val = row.get('avg_unit_price', 15.0)
            p_id = get_product_id(conn, p_name, c_name, price_val)
            
            forecast_records.append({
                "product_id": p_id,
                "forecast_date": str(row['date']).split(' ')[0],
                "predicted_demand": float(row['predicted_demand']),
                "lower_bound": float(row['predicted_demand'] - (row.get('rolling_7_std', 0.0) * 1.96)),
                "upper_bound": float(row['predicted_demand'] + (row.get('rolling_7_std', 0.0) * 1.96)),
                "actual_demand": float(row['quantity_sold']) if 'quantity_sold' in row else None,
                "lag_1": float(row.get('lag_1', 0.0)),
                "lag_7": float(row.get('lag_7', 0.0)),
                "lag_14": float(row.get('lag_14', 0.0)),
                "rolling_7_mean": float(row.get('rolling_7_mean', 0.0)),
                "rolling_14_mean": float(row.get('rolling_14_mean', 0.0)),
                "rolling_7_std": float(row.get('rolling_7_std', 0.0)),
                "promotion_flag": int(row.get('promotion_flag', 0)),
                "festival_flag": int(row.get('festival_flag', 0))
            })
        save_forecasts(conn, model_id, forecast_records)
        
        # Query inserted forecast ids to link recommendations
        forecast_ids = {}
        for row in conn.execute("SELECT forecast_id, product_id, forecast_date FROM demand_forecasts WHERE model_id = ?", (model_id,)).fetchall():
            forecast_ids[(row["product_id"], row["forecast_date"])] = row["forecast_id"]
            
        # 2. Insert recommendations
        recommendation_records = []
        for idx, row in df.iterrows():
            p_name = row['product_name']
            p_id = get_product_id(conn, p_name)
            f_date = str(row['date']).split(' ')[0]
            f_id = forecast_ids.get((p_id, f_date))
            
            if f_id:
                recommendation_records.append({
                    "forecast_id": f_id,
                    "product_id": p_id,
                    "recommendation_date": f_date,
                    "net_stock": float(row['net_stock']),
                    "predicted_demand": float(row['predicted_demand']),
                    "reorder_quantity": float(row['safety_adjusted_reorder_qty']),
                    "inventory_status": row['inventory_status'],
                    "risk_tier": row['stock_risk_level'].replace(' Risk', ''),
                    "stockout_probability": float(row['stockout_probability'])
                })
        save_inventory_recommendations(conn, recommendation_records)
        
        # 3. Insert alerts
        for idx, row in df.iterrows():
            if row['alert_severity'] in ['Critical', 'Warning']:
                p_name = row['product_name']
                p_id = get_product_id(conn, p_name)
                f_date = str(row['date']).split(' ')[0]
                
                alert_map = {
                    "Stockout": "stockout",
                    "Understocked": "stockout",
                    "Spoilage Risk": "spoilage_risk",
                    "Abnormal Demand": "abnormal_demand",
                    "Overstock": "overstock"
                }
                atype = alert_map.get(row['alert_type'], "stockout")
                
                save_alert(
                    conn, 
                    p_id, 
                    f_date, 
                    atype, 
                    row['alert_severity'], 
                    row['alert_message'], 
                    row['recommended_action'],
                    float(row['net_stock']), 
                    float(row['predicted_demand'])
                )
    except Exception as e:
        print(f"[Pipeline SQL Error] Failed to write outputs to DB: {str(e)}")
    finally:
        conn.close()
        
    return df
