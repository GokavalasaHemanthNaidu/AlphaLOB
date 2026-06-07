import pytest
import numpy as np
from src.domain.models.regime_hmm import RegimeHMM
import os

def test_regime_hmm_initialization():
    model = RegimeHMM(n_states=3)
    assert model.n_states == 3
    assert not model.is_fitted

def test_regime_hmm_fit_and_predict():
    model = RegimeHMM(n_states=3)
    
    # Create synthetic data with 3 distinct regimes of volatility (feature index 0)
    # Regime 0: Low vol (Mean Reverting)
    # Regime 1: Medium vol (Trending)
    # Regime 2: High vol (Volatile)
    
    np.random.seed(42)
    low_vol = np.random.normal(0.01, 0.005, (100, 2))
    med_vol = np.random.normal(0.05, 0.01, (100, 2))
    high_vol = np.random.normal(0.20, 0.05, (100, 2))
    
    features = np.vstack([low_vol, med_vol, high_vol])
    
    # Fit the model
    model.fit(features)
    
    assert model.is_fitted
    
    # Test predictions
    labels = model.predict_regime_labels(features)
    
    assert len(labels) == 300
    assert "MEAN_REV" in labels
    assert "TRENDING" in labels
    assert "VOLATILE" in labels

def test_regime_hmm_save_load(tmp_path):
    model = RegimeHMM(n_states=3)
    features = np.random.normal(0.05, 0.01, (50, 2))
    model.fit(features)
    
    filepath = tmp_path / "test_regime_hmm.bin"
    model.save(str(filepath))
    
    assert os.path.exists(filepath)
    
    loaded_model = RegimeHMM.load(str(filepath))
    assert loaded_model.is_fitted
    assert loaded_model.n_states == 3
    assert loaded_model.state_labels == model.state_labels
