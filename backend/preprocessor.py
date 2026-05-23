import pandas as pd
import numpy as np

from backend.schema_validator import detect_dataset_type

MODEL_FEATURES = [
    'day_of_week', 'month', 'quarter', 'week_of_year', 'weekend_flag',
    'festival_flag', 'lag_1', 'lag_7', 'lag_14', 'rolling_7_mean',
    'rolling_14_mean', 'rolling_7_std', 'promotion_flag', 'campaign_active',
    'marketing_intensity', 'avg_roas', 'stock_turnover_rate',
    'avg_unit_price', 'season_Autumn', 'season_Monsoon', 'season_Summer',
    'season_Winter', 'category_Cold Drinks & Juices',
    'category_Hot Beverages', 'category_Snacks & Munchies'
]

def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Autumn'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame by parsing dates, renaming columns, and f-filling optional cols.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Rename columns to map to training schema
    column_mapping = {
        'inventory_stock': 'net_stock',
        'price': 'avg_unit_price'
    }
    df = df.rename(columns=column_mapping)

    if detect_dataset_type(df) != "ml_feature_matrix":
        df = df.drop_duplicates(subset=['date', 'product_name'])
    
    # Handle optional columns forward fill
    optional_cols = ['promotion_flag', 'weather_temp', 'region']
    for col in optional_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)
            
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rolling time series features and calendar properties.
    Skips recomputation when the upload is already an ML feature matrix.
    """
    df = df.copy()

    if detect_dataset_type(df) == "ml_feature_matrix":
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.dayofweek
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "quarter" not in df.columns:
            df["quarter"] = df["date"].dt.quarter
        if "week_of_year" not in df.columns:
            df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        if "weekend_flag" not in df.columns:
            df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)
        if "season" not in df.columns and "month" in df.columns:
            df["season"] = df["month"].apply(get_season)
        return df

    # Sort chronologically per product for correct time series feature computation
    df = df.sort_values(by=['product_name', 'date']).reset_index(drop=True)

    # Compute lags on quantity_sold
    df['lag_1'] = df.groupby('product_name')['quantity_sold'].transform(lambda x: x.shift(1)).fillna(0.0)
    df['lag_7'] = df.groupby('product_name')['quantity_sold'].transform(lambda x: x.shift(7)).fillna(0.0)
    df['lag_14'] = df.groupby('product_name')['quantity_sold'].transform(lambda x: x.shift(14)).fillna(0.0)
    
    # Compute rolling windows on quantity_sold (shift 1 to prevent lookahead bias)
    df['rolling_7_mean'] = df.groupby('product_name')['quantity_sold'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()).fillna(0.0)
    df['rolling_14_mean'] = df.groupby('product_name')['quantity_sold'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(0.0)
    df['rolling_7_std'] = df.groupby('product_name')['quantity_sold'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std()).fillna(0.0)
    
    # Calendar variables
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['weekend_flag'] = (df['day_of_week'] >= 5).astype(int)
    
    # Missing optional model features defaults
    if 'festival_flag' not in df.columns:
        df['festival_flag'] = 0
    if 'promotion_flag' not in df.columns:
        df['promotion_flag'] = 0
    if 'campaign_active' not in df.columns:
        df['campaign_active'] = 0
    if 'marketing_intensity' not in df.columns:
        df['marketing_intensity'] = 0.0
    if 'avg_roas' not in df.columns:
        df['avg_roas'] = 0.0
        
    # Calculate stock turnover rate if not present
    if 'stock_turnover_rate' not in df.columns:
        df['stock_turnover_rate'] = df['quantity_sold'] / (df['net_stock'] + 1e-5)
        
    # Standard season string derivation
    df['season'] = df['month'].apply(get_season)
    
    return df

def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates dummy columns for category and season and forces int representation.
    """
    df = df.copy()
    
    # Drop pre-existing OHE columns to prevent duplicate columns during get_dummies
    ohe_cols = [c for c in df.columns if c.startswith('category_') or c.startswith('season_')]
    df = df.drop(columns=ohe_cols, errors='ignore')
    
    # Generate OHE columns for category and season, but preserve original category
    df['category_orig'] = df['category']
    df = pd.get_dummies(df, columns=['category', 'season'], prefix=['category', 'season'])
    df = df.rename(columns={'category_orig': 'category'})
    
    # Remove duplicate columns if they somehow still exist
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Convert all boolean columns to int
    for col in df.columns:
        if hasattr(df[col], 'dtype') and (df[col].dtype == bool or df[col].dtype == 'bool'):
            df[col] = df[col].astype(int)
            
    return df

def align_to_model_features(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Guarantees the DataFrame features align precisely with target model inputs.
    """
    df = df.copy()
    
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0
            
    # Keep metadata columns as well as features
    metadata_cols = ['date', 'product_name', 'quantity_sold', 'net_stock', 'category', 'avg_unit_price']
    existing_meta = [col for col in metadata_cols if col in df.columns]
    
    # Return dataframe with metadata followed by aligned features
    # Ensure there are no duplicate columns
    seen = set()
    all_cols = []
    for col in (existing_meta + model_features):
        if col not in seen:
            all_cols.append(col)
            seen.add(col)
            
    return df[all_cols]

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes raw-to-processed pipeline.
    """
    df = clean_data(df)
    df = feature_engineer(df)
    df = encode_categories(df)
    df = align_to_model_features(df, MODEL_FEATURES)
    return df
