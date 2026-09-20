import os
import joblib
import numpy as np
import pytest

MODEL_PATH = "best_model.pkl"

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)

def test_model_loaded(model):
    assert model is not None
    assert hasattr(model, "predict"), "Model must implement predict method"
    assert hasattr(model, "predict_proba"), "Model must implement predict_proba method"

def test_model_feature_expectation(model):
    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == 16, f"Expected 16 features, got {model.n_features_in_}"

def test_model_single_prediction_shape(model):
    # Dummy feature vector with 16 dimensions
    dummy_input = np.zeros((1, 16))
    pred = model.predict(dummy_input)
    proba = model.predict_proba(dummy_input)
    
    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert np.isclose(np.sum(proba), 1.0, atol=1e-4)

def test_model_batch_prediction_shape(model):
    dummy_batch = np.random.randn(10, 16)
    preds = model.predict(dummy_batch)
    probas = model.predict_proba(dummy_batch)
    
    assert len(preds) == 10
    assert probas.shape == (10, 2)
    for p in preds:
        assert p in [0, 1]
