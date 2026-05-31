import torch
from src.domain.models.multi_task_head import MultiTaskLOBModel

def test_lob_transformer_forward_pass():
    """
    Test that the MultiTaskLOBModel accepts the correct tensor shapes
    and outputs the expected dictionary of predictions.
    """
    # Batch size of 2, 10 levels, 2 features (price, volume)
    batch_size = 2
    num_levels = 10
    
    # Create dummy tensors for bids and asks
    dummy_bids = torch.randn(batch_size, num_levels, 2)
    dummy_asks = torch.randn(batch_size, num_levels, 2)
    
    # Initialize model
    model = MultiTaskLOBModel(num_levels=num_levels, d_model=64)
    
    # Set to evaluation mode to disable dropout
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_bids, dummy_asks)
        
    # Check output structure and shapes
    assert "direction_logits" in outputs
    assert "spread_prob" in outputs
    assert "imbalance_pred" in outputs
    
    assert outputs["direction_logits"].shape == (batch_size, 3)
    assert outputs["spread_prob"].shape == (batch_size, 1)
    assert outputs["imbalance_pred"].shape == (batch_size, 1)
    
    # Verify probability is between 0 and 1
    assert (outputs["spread_prob"] >= 0).all() and (outputs["spread_prob"] <= 1).all()
