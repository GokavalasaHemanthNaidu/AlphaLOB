import pytest
from src.domain.features import FeatureEngine

def test_feature_engine_initialization():
    engine = FeatureEngine(depth_levels=10)
    assert engine.depth_levels == 10
    assert engine.prev_bids == {}
    assert engine.prev_asks == {}

def test_spread_and_depth_calculation():
    engine = FeatureEngine()
    bids = {100.0: 1.5, 99.0: 2.0}
    asks = {101.0: 0.5, 102.0: 3.0}
    
    spread, bid_depth, ask_depth = engine._calculate_spread_and_depth(bids, asks)
    
    assert spread == 1.0  # 101.0 - 100.0
    assert bid_depth == 3.5 # 1.5 + 2.0
    assert ask_depth == 3.5 # 0.5 + 3.0

def test_wofi_calculation_initial_state():
    engine = FeatureEngine()
    bids = {100.0: 1.5, 99.0: 2.0}
    asks = {101.0: 0.5, 102.0: 3.0}
    
    wofi = engine._calculate_wofi(bids, asks)
    assert wofi == 0.0
    assert engine.prev_bids == bids
    assert engine.prev_asks == asks

def test_wofi_calculation_update():
    engine = FeatureEngine()
    bids_t1 = {100.0: 1.0}
    asks_t1 = {101.0: 1.0}
    engine._calculate_wofi(bids_t1, asks_t1) # sets prev state
    
    # Increase bid size at same price
    bids_t2 = {100.0: 2.0}
    asks_t2 = {101.0: 1.0}
    wofi = engine._calculate_wofi(bids_t2, asks_t2)
    
    # math.exp(-0.5 * 0) = 1.0 weight. size change is +1.0. wofi should be 1.0
    assert wofi > 0.0

def test_process_empty_snapshot():
    engine = FeatureEngine()
    snapshot = {"b": [], "a": [], "ts": 1234567890}
    features = engine.process(snapshot)
    assert features == {}

def test_process_valid_snapshot():
    engine = FeatureEngine()
    snapshot = {
        "b": [["100.0", "1.5"], ["99.0", "2.0"]],
        "a": [["101.0", "0.5"], ["102.0", "3.0"]],
        "ts": 1234567890
    }
    
    features = engine.process(snapshot)
    
    assert features["spread"] == 1.0
    assert features["bid_depth"] == 3.5
    assert features["ask_depth"] == 3.5
    assert features["wofi"] == 0.0
