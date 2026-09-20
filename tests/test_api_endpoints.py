import pytest
import httpx
from app import app

@pytest.fixture
def sample_stock_record():
    return {
        "Open": 2500.0, "High": 2550.0, "Low": 2490.0, "Close": 2540.0,
        "AdjClose": 2540.0, "Volume": 4500000.0, "Daily_Return": 0.016,
        "MA_20": 2480.0, "MA_50": 2420.0, "Volatility_20D": 0.015,
        "RSI": 62.5, "MACD": 14.8, "Return_Lag_1": 0.012,
        "Return_Lag_2": 0.005, "Return_Lag_5": 0.021, "Volume_Change": 0.085
    }

@pytest.mark.asyncio
async def test_health_endpoint():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["feature_count"] == 16

@pytest.mark.asyncio
async def test_predict_endpoint_valid_payload(sample_stock_record):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict", json=sample_stock_record)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["prediction_label"] in ["UP", "DOWN"]
        assert data["prediction_class"] in [0, 1]
        assert 0.0 <= data["confidence_score"] <= 1.0

@pytest.mark.asyncio
async def test_predict_batch_endpoint(sample_stock_record):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {"stocks": [sample_stock_record, sample_stock_record]}
        resp = await client.post("/predict_batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 2
        assert len(data["predictions"]) == 2

@pytest.mark.asyncio
async def test_predict_negative_missing_features():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        incomplete_payload = {"Open": 2500.0, "Close": 2540.0}
        resp = await client.post("/predict", json=incomplete_payload)
        assert resp.status_code == 422
