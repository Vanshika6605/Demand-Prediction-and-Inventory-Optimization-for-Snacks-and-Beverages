from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os
import sys

# Add root folder to sys path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import predict_demand, generate_inventory_alerts, generate_reorder_recommendations
from rag.rag_engine import RAGEngine

app = FastAPI(
    title="AI-Powered FMCG Demand Forecaster & Inventory Optimizer API",
    description="Production-grade supply chain intelligence API with gradient boosting pipelines and RAG explainability.",
    version="1.0.0"
)

# -------------------------------------------------------------
# PYDANTIC DATA SCHEMAS
# -------------------------------------------------------------
class PredictRequest(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    month: int = Field(..., ge=1, le=12, description="1 to 12")
    quarter: int = Field(..., ge=1, le=4, description="1 to 4")
    week_of_year: int = Field(..., ge=1, le=53, description="1 to 53")
    weekend_flag: int = Field(..., ge=0, le=1, description="1 if weekend, else 0")
    festival_flag: int = Field(..., ge=0, le=1, description="1 if holiday, else 0")
    lag_1: float = Field(..., ge=0.0, description="Units sold 1 day ago")
    lag_7: float = Field(..., ge=0.0, description="Units sold 7 days ago")
    lag_14: float = Field(..., ge=0.0, description="Units sold 14 days ago")
    rolling_7_mean: float = Field(..., ge=0.0, description="7-day demand moving average")
    rolling_14_mean: float = Field(..., ge=0.0, description="14-day demand moving average")
    rolling_7_std: float = Field(..., ge=0.0, description="7-day demand standard deviation")
    promotion_flag: int = Field(..., ge=0, le=1, description="1 if promo active, else 0")
    campaign_active: int = Field(..., ge=0, description="Active marketing campaign code")
    marketing_intensity: float = Field(..., ge=0.0, le=1.0, description="Marketing budget strength scale 0 to 1")
    avg_roas: float = Field(..., ge=0.0, description="Average Return on Ad Spend")
    stock_turnover_rate: float = Field(..., ge=0.0, description="Stock turnover frequency")
    avg_unit_price: float = Field(..., ge=0.0, description="FMCG unit price")
    season_Autumn: int = Field(0, ge=0, le=1)
    season_Monsoon: int = Field(0, ge=0, le=1)
    season_Summer: int = Field(0, ge=0, le=1)
    season_Winter: int = Field(0, ge=0, le=1)
    category_Cold_Drinks_Juices: int = Field(0, ge=0, le=1, alias="category_Cold Drinks & Juices")
    category_Hot_Beverages: int = Field(0, ge=0, le=1, alias="category_Hot Beverages")
    category_Snacks_Munchies: int = Field(0, ge=0, le=1, alias="category_Snacks & Munchies")

    class Config:
        populate_by_name = True

class PredictResponse(BaseModel):
    predicted_demand: float
    model_used: str

class AlertRequest(BaseModel):
    predicted_demand: float = Field(..., ge=0.0)
    net_stock: float = Field(..., ge=0.0)

class AlertResponse(BaseModel):
    inventory_status: str
    stock_risk_level: str
    alert_severity: str
    stockout_probability: float
    risk_color: str
    severity_color: str

class ReorderRequest(BaseModel):
    predicted_demand: float = Field(..., ge=0.0)
    net_stock: float = Field(..., ge=0.0)

class ReorderResponse(BaseModel):
    reorder_required: bool
    safety_adjusted_reorder_qty: int

class RAGRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language search question")
    k: Optional[int] = Field(3, ge=1, le=10, description="Top K insights to retrieve")

class RAGResponseItem(BaseModel):
    document: str
    score: float
    metadata: Dict

# -------------------------------------------------------------
# RAG ENGINE INITIALIZATION
# -------------------------------------------------------------
rag_instance = None

def get_rag_engine():
    global rag_instance
    if rag_instance is None:
        csv_path = os.path.join("outputs", "inventory_recommendations.csv")
        if not os.path.exists(csv_path):
            # Create a mock or default output folder and write placeholder if needed
            print(f"[API] Warning: Output recommendations CSV not found at {csv_path}. Please run train_and_serialize.py first.")
        rag_instance = RAGEngine(data_path=csv_path)
    return rag_instance

# -------------------------------------------------------------
# ENDPOINT CONTROLLERS
# -------------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "Healthy",
        "service": "FMCG Demand Forecaster API Backend",
        "deployment_ready": True
    }

@app.post("/predict", response_model=List[PredictResponse])
def api_predict_demand(payload: List[PredictRequest]):
    try:
        results = []
        model_path = os.path.join("models", "xgboost_model.pkl")
        for item in payload:
            features = item.dict(by_alias=True)
            pred = predict_demand(features, model_path)
            results.append(PredictResponse(
                predicted_demand=round(pred, 2),
                model_used="XGBoost Regressor (Champion)"
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inventory-alerts", response_model=AlertResponse)
def api_inventory_alerts(payload: AlertRequest):
    try:
        alerts = generate_inventory_alerts(payload.predicted_demand, payload.net_stock)
        return AlertResponse(**alerts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reorder-recommendations", response_model=ReorderResponse)
def api_reorder_recommendations(payload: ReorderRequest):
    try:
        recs = generate_reorder_recommendations(payload.predicted_demand, payload.net_stock)
        return ReorderResponse(**recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast-insights", response_model=List[RAGResponseItem])
def api_forecast_insights(payload: RAGRequest):
    try:
        engine = get_rag_engine()
        search_results = engine.search(payload.query, payload.k)
        
        response_items = []
        for res in search_results:
            response_items.append(RAGResponseItem(
                document=res["document"],
                score=res["score"],
                metadata=res["metadata"]
            ))
        return response_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
