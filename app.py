import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="NIFTY 50 Stock Direction Prediction API",
    description="Production API delivering real-time stock directional forecasts (UP / DOWN) using serialized Machine Learning models.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Warning: Model could not be loaded from {MODEL_PATH}: {e}")

FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume',
    'Daily_Return', 'MA_20', 'MA_50', 'Volatility_20D', 'RSI', 'MACD',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_5', 'Volume_Change'
]

class StockFeatures(BaseModel):
    Open: float = Field(..., description="Opening price of the stock", example=2450.0)
    High: float = Field(..., description="Daily high price", example=2485.5)
    Low: float = Field(..., description="Daily low price", example=2430.0)
    Close: float = Field(..., description="Closing price", example=2475.2)
    AdjClose: float = Field(..., description="Adjusted closing price", example=2475.2)
    Volume: float = Field(..., description="Trading volume", example=5820000.0)
    Daily_Return: float = Field(..., description="Daily percentage return", example=0.0125)
    MA_20: float = Field(..., description="20-day moving average", example=2420.0)
    MA_50: float = Field(..., description="50-day moving average", example=2380.0)
    Volatility_20D: float = Field(..., description="20-day rolling annualized volatility", example=0.018)
    RSI: float = Field(..., description="14-day Relative Strength Index", example=58.4)
    MACD: float = Field(..., description="Moving Average Convergence Divergence", example=12.5)
    Return_Lag_1: float = Field(..., description="Return lagged by 1 trading session", example=0.008)
    Return_Lag_2: float = Field(..., description="Return lagged by 2 trading sessions", example=-0.004)
    Return_Lag_5: float = Field(..., description="Return lagged by 5 trading sessions", example=0.015)
    Volume_Change: float = Field(..., description="Percentage change in volume", example=0.052)

class BatchStockFeatures(BaseModel):
    stocks: List[StockFeatures]

class PredictionResponse(BaseModel):
    status: str
    prediction_label: str
    prediction_class: int
    probability_up: float
    probability_down: float
    confidence_score: float

class BatchPredictionResponse(BaseModel):
    status: str
    total_samples: int
    predictions: List[PredictionResponse]

@app.get("/", tags=["Health & Info"])
def root() -> Dict[str, Any]:
    return {
        "service": "NIFTY 50 Stock Direction Prediction Service",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", tags=["Health & Info"])
def health() -> Dict[str, Any]:
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model artifact is not loaded."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(FEATURE_COLUMNS),
        "model_type": type(model).__name__
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: StockFeatures) -> PredictionResponse:
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    data_dict = (features.model_dump() if hasattr(features, "model_dump") else features.dict())
    feature_vector = pd.DataFrame([data_dict])[FEATURE_COLUMNS]

    try:
        pred_class = int(model.predict(feature_vector)[0])
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            prob_down = float(probs[0])
            prob_up = float(probs[1])
        else:
            prob_up = 1.0 if pred_class == 1 else 0.0
            prob_down = 1.0 - prob_up

        label = "UP" if pred_class == 1 else "DOWN"
        confidence = max(prob_up, prob_down)

        return PredictionResponse(
            status="success",
            prediction_label=label,
            prediction_class=pred_class,
            probability_up=round(prob_up, 4),
            probability_down=round(prob_down, 4),
            confidence_score=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchStockFeatures) -> BatchPredictionResponse:
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    if not batch.stocks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least one record."
        )

    rows = [(s.model_dump() if hasattr(s, "model_dump") else s.dict()) for s in batch.stocks]
    feature_df = pd.DataFrame(rows)[FEATURE_COLUMNS]

    try:
        preds = model.predict(feature_df)
        has_proba = hasattr(model, "predict_proba")
        probs = model.predict_proba(feature_df) if has_proba else None

        results = []
        for i in range(len(preds)):
            pred_class = int(preds[i])
            if has_proba:
                p_down = float(probs[i][0])
                p_up = float(probs[i][1])
            else:
                p_up = 1.0 if pred_class == 1 else 0.0
                p_down = 1.0 - p_up

            label = "UP" if pred_class == 1 else "DOWN"
            confidence = max(p_up, p_down)

            results.append(PredictionResponse(
                status="success",
                prediction_label=label,
                prediction_class=pred_class,
                probability_up=round(p_up, 4),
                probability_down=round(p_down, 4),
                confidence_score=round(confidence, 4)
            ))

        return BatchPredictionResponse(
            status="success",
            total_samples=len(results),
            predictions=results
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
