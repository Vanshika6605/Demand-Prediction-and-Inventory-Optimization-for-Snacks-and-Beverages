import streamlit as st
import pandas as pd

def render_filters(df: pd.DataFrame, key_prefix="") -> pd.DataFrame:
    """
    Renders a responsive 4-column filter row (Product, Category, Date Range, Stock Status).
    Returns the filtered DataFrame subset.
    """
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        products = st.multiselect(
            "Target Product Focus", 
            options=sorted(df['product_name'].unique().tolist()) if 'product_name' in df.columns else [], 
            key=f"{key_prefix}_prod"
        )
    with col2:
        categories = st.multiselect(
            "Category Filter", 
            options=sorted(df['category'].unique().tolist()) if 'category' in df.columns else [], 
            key=f"{key_prefix}_cat"
        )
    with col3:
        if 'date' in df.columns and not df.empty:
            min_date = df['date'].min()
            max_date = df['date'].max()
            date_range = st.date_input(
                "Operation Horizon", 
                value=(min_date, max_date),
                key=f"{key_prefix}_date"
            )
        else:
            date_range = st.date_input(
                "Operation Horizon", 
                value=pd.Timestamp.now(),
                key=f"{key_prefix}_date"
            )
    with col4:
        status_options = sorted(df['inventory_status'].unique().tolist()) if 'inventory_status' in df.columns else ["Normal", "Understocked", "Overstocked"]
        status = st.multiselect(
            "Stock Status", 
            options=status_options, 
            key=f"{key_prefix}_status"
        )
        
    # Apply filters
    if products:
        df = df[df['product_name'].isin(products)]
    if categories:
        df = df[df['category'].isin(categories)]
        
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        df = df[(df['date'] >= pd.Timestamp(date_range[0])) & (df['date'] <= pd.Timestamp(date_range[1]))]
        
    if status:
        df = df[df['inventory_status'].isin(status)]
        
    return df
