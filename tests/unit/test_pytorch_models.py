import pytest
import sys
import numpy as np
from unittest.mock import MagicMock

# We use importorskip because PyTorch is a heavy dependency meant for Google Colab.
# If run on the laptop without PyTorch, it gracefully skips these tests instead of failing.
torch = pytest.importorskip("torch")

# Mock hmmlearn globally before importing
mock_hmm = MagicMock()
mock_hmm.GaussianHMM.return_value.predict.return_value = np.zeros(100)
mock_hmm.GaussianHMM.return_value.means_ = np.array([[0.1], [0.2], [0.3]])
sys.modules['hmmlearn'] = mock_hmm
sys.modules['hmmlearn.hmm'] = mock_hmm

from src.domain.models.lob_transformer import LOBTransformer
from src.domain.models.multi_task_head import AlphaLOBModel, MultiTaskHead
from src.domain.models.regime_hmm import RegimeHMM

def test_lob_transformer_shape():
    batch_size = 4
    n_levels = 10
    features = 4
    d_model = 64
    
    model = LOBTransformer(n_levels=n_levels, features_per_level=features, d_model=d_model)
    dummy_input = torch.randn(batch_size, n_levels, features)
    
    output = model(dummy_input)
    
    # Expected output: flattened across n_levels
    # [batch_size, n_levels * d_model]
    assert output.shape == (batch_size, n_levels * d_model)

def test_alpha_lob_multi_task_shape():
    batch_size = 4
    model = AlphaLOBModel(n_levels=10, features_per_level=4, d_model=64)
    dummy_input = torch.randn(batch_size, 10, 4)
    
    dir_5s, dir_30s, dir_5m, spread, vol = model(dummy_input)
    
    assert dir_5s.shape == (batch_size, 1)
    assert dir_30s.shape == (batch_size, 1)
    assert dir_5m.shape == (batch_size, 1)
    assert spread.shape == (batch_size, 1)
    assert vol.shape == (batch_size, 1)

def test_regime_hmm_mocked():
    # hmmlearn might also not be installed locally
    hmmlearn = pytest.importorskip("hmmlearn")
    
    hmm = RegimeHMM(n_states=3)
    features = np.random.randn(100, 2)
    
    hmm.fit(features)
    assert hmm.is_fitted
    
    predictions = hmm.predict(features)
    # If mocked, it might return a MagicMock, otherwise a numpy array
    # We just ensure it doesn't crash
    assert predictions is not None
