import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from backend.preprocessor import MODEL_FEATURES

def train_and_select(df_processed: pd.DataFrame, upload_id: int = None, user_id: int = None) -> dict:
    """
    Trains XGBoost, LightGBM, and CatBoost models on processed data,
    evaluates them chronologically, updates the serialized pkl binaries,
    logs the champion model to fmcg_platform.db, and returns metrics.
    """
    # Chronological sort and split (80/20)
    df_sorted = df_processed.sort_values(by='date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    target_col = 'quantity_sold'
    
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[target_col]
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[target_col]
    
    # 1. XGBoost
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
    
    # 2. LightGBM
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
    
    # 3. CatBoost
    cat_model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_lgb = lgb_model.predict(X_test)
    y_pred_cat = cat_model.predict(X_test)
    
    metrics = {
        'XGBoost': {
            'MAE': float(mean_absolute_error(y_test, y_pred_xgb)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
            'R2': float(r2_score(y_test, y_pred_xgb))
        },
        'LightGBM': {
            'MAE': float(mean_absolute_error(y_test, y_pred_lgb)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_lgb))),
            'R2': float(r2_score(y_test, y_pred_lgb))
        },
        'CatBoost': {
            'MAE': float(mean_absolute_error(y_test, y_pred_cat)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_cat))),
            'R2': float(r2_score(y_test, y_pred_cat))
        }
    }
    
    # Serialize models
    os.makedirs("models", exist_ok=True)
    xgb_path = os.path.join("models", "xgboost_model.pkl")
    lgb_path = os.path.join("models", "lightgbm_model.pkl")
    cat_path = os.path.join("models", "catboost_model.pkl")
    
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgb_model, lgb_path)
    joblib.dump(cat_model, cat_path)
    
    # Log champion (XGBoost) model to registry in SQLite
    from database_setup import get_connection, log_model
    conn = get_connection()
    try:
        db_metrics = {
            "r2": metrics['XGBoost']['R2'],
            "mae": metrics['XGBoost']['MAE'],
            "rmse": metrics['XGBoost']['RMSE'],
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "feature_count": len(MODEL_FEATURES)
        }
        model_id = log_model(
            conn, 
            upload_id=upload_id or 1, 
            model_type="xgboost", 
            metrics=db_metrics, 
            model_path=xgb_path, 
            trained_by=user_id or 1
        )
        metrics['champion_model_id'] = model_id
    except Exception as e:
        print(f"[Trainer DB Error] Failed to log model: {str(e)}")
    finally:
        conn.close()
        
    return metrics
