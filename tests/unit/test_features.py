import pytest
from src.domain.features import FeatureEngine

def test_feature_engine_spread_and_depth():
    engine = FeatureEngine(depth_levels=2)
    
    # Mock snapshot
    snapshot = {
        "b": [["60000.0", "2.0"], ["59999.0", "1.0"]],
        "a": [["60001.0", "1.5"], ["60002.0", "2.5"]],
        "ts": 123456789
    }
    
    features = engine.process(snapshot)
    
    assert features["spread"] == 1.0  # 60001.0 - 60000.0
    assert features["bid_depth"] == 3.0 # 2.0 + 1.0
    assert features["ask_depth"] == 4.0 # 1.5 + 2.5

def test_feature_engine_wofi():
    engine = FeatureEngine(depth_levels=2)
    
    snapshot_1 = {
        "b": [["60000.0", "2.0"]],
        "a": [["60001.0", "1.5"]],
        "ts": 1
    }
    
    # First snapshot shouldn't have WOFI because there's no previous state
    features_1 = engine.process(snapshot_1)
    assert features_1["wofi"] == 0.0
    
    # Second snapshot: Bid volume increases at best bid (positive WOFI)
    snapshot_2 = {
        "b": [["60000.0", "3.0"]],
        "a": [["60001.0", "1.5"]],
        "ts": 2
    }
    
    features_2 = engine.process(snapshot_2)
    # The weight for level 0 is exp(0) = 1.0. The size diff is 1.0. 
    # WOFI = 1.0 * 1.0 = 1.0
    assert features_2["wofi"] > 0.0
    
    # Third snapshot: Ask volume increases at best ask (negative WOFI)
    snapshot_3 = {
        "b": [["60000.0", "3.0"]],
        "a": [["60001.0", "5.0"]],
        "ts": 3
    }
    features_3 = engine.process(snapshot_3)
    assert features_3["wofi"] < 0.0
