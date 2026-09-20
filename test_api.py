import sys
import json
import asyncio
import httpx
from app import app

BASE_URL = "http://127.0.0.1:8000"

# Sample payloads
sample_bullish = {
    "Open": 2500.0,
    "High": 2550.0,
    "Low": 2490.0,
    "Close": 2540.0,
    "AdjClose": 2540.0,
    "Volume": 4500000.0,
    "Daily_Return": 0.016,
    "MA_20": 2480.0,
    "MA_50": 2420.0,
    "Volatility_20D": 0.015,
    "RSI": 62.5,
    "MACD": 14.8,
    "Return_Lag_1": 0.012,
    "Return_Lag_2": 0.005,
    "Return_Lag_5": 0.021,
    "Volume_Change": 0.085
}

sample_bearish = {
    "Open": 2500.0,
    "High": 2505.0,
    "Low": 2430.0,
    "Close": 2435.0,
    "AdjClose": 2435.0,
    "Volume": 6500000.0,
    "Daily_Return": -0.026,
    "MA_20": 2510.0,
    "MA_50": 2540.0,
    "Volatility_20D": 0.028,
    "RSI": 34.0,
    "MACD": -16.5,
    "Return_Lag_1": -0.018,
    "Return_Lag_2": -0.008,
    "Return_Lag_5": -0.035,
    "Volume_Change": 0.210
}

async def run_tests_async():
    print("=" * 65)
    print("EXPERIMENT 6: LOCAL API INFERENCE & CONTRACT VERIFICATION")
    print("=" * 65)
    
    # We use ASGITransport which simulates the exact ASGI HTTP request cycle
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Healthcheck Endpoint Test
        try:
            print("\n[Test 1] Querying GET /health ...")
            resp = await client.get("/health")
            print(f"HTTP Status: {resp.status_code}")
            print("Response:", json.dumps(resp.json(), indent=2))
            assert resp.status_code == 200, "Health check failed!"
            print(">>> Test 1 PASSED: Model service is online and healthy.")
        except Exception as e:
            print(f"Test 1 Failed: {e}")
            return False

        # 2. Bullish Prediction Test
        try:
            print("\n[Test 2] Querying POST /predict with sample payload (Bullish Scenario)...")
            resp = await client.post("/predict", json=sample_bullish)
            print(f"HTTP Status: {resp.status_code}")
            print("Response:", json.dumps(resp.json(), indent=2))
            assert resp.status_code == 200, "Prediction failed!"
            data = resp.json()
            assert "prediction_label" in data and data["prediction_label"] in ["UP", "DOWN"]
            print(f">>> Test 2 PASSED: Received valid prediction: {data['prediction_label']} (Confidence: {data['confidence_score']})")
        except Exception as e:
            print(f"Test 2 Failed: {e}")
            return False

        # 3. Bearish Prediction Test
        try:
            print("\n[Test 3] Querying POST /predict with sample payload (Bearish Scenario)...")
            resp = await client.post("/predict", json=sample_bearish)
            print(f"HTTP Status: {resp.status_code}")
            print("Response:", json.dumps(resp.json(), indent=2))
            assert resp.status_code == 200, "Prediction failed!"
            data = resp.json()
            assert "prediction_label" in data
            print(f">>> Test 3 PASSED: Received valid prediction: {data['prediction_label']} (Confidence: {data['confidence_score']})")
        except Exception as e:
            print(f"Test 3 Failed: {e}")
            return False

        # 4. Batch Prediction Test
        try:
            print("\n[Test 4] Querying POST /predict_batch with multiple records...")
            batch_payload = {"stocks": [sample_bullish, sample_bearish]}
            resp = await client.post("/predict_batch", json=batch_payload)
            print(f"HTTP Status: {resp.status_code}")
            print(f"Batch processed count: {resp.json().get('total_samples')}")
            assert resp.status_code == 200, "Batch prediction failed!"
            print(">>> Test 4 PASSED: Batch inference processed correctly.")
        except Exception as e:
            print(f"Test 4 Failed: {e}")
            return False

        # 5. Schema Validation & Error Handling Test (Negative Testing)
        try:
            print("\n[Test 5] Querying POST /predict with missing required fields (Schema Validation)...")
            invalid_payload = {"Open": 2500.0, "Close": 2540.0} # Missing 14 features
            resp = await client.post("/predict", json=invalid_payload)
            print(f"HTTP Status: {resp.status_code} (Expected 422 Unprocessable Entity)")
            assert resp.status_code == 422, f"Expected status 422, got {resp.status_code}"
            print(">>> Test 5 PASSED: Pydantic correctly blocked invalid request with status 422.")
        except Exception as e:
            print(f"Test 5 Failed: {e}")
            return False

    print("\n" + "=" * 65)
    print("ALL EXPERIMENT 6 API TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True

def run_tests():
    return asyncio.run(run_tests_async())

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
