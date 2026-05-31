import os
import torch
import torch.nn as nn
from src.domain.models.multi_task_head import MultiTaskLOBModel

def generate_dummy_onnx(output_path="models/lob_transformer.onnx"):
    print("Generating dummy ONNX model...")
    # Initialize the model with our expected dimensions (10 levels, d_model=64)
    model = MultiTaskLOBModel(num_levels=10, d_model=64)
    model.eval()

    # Create dummy input tensors (batch_size=1, num_levels=10, features=2)
    dummy_bids = torch.randn(1, 10, 2)
    dummy_asks = torch.randn(1, 10, 2)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    torch.onnx.export(
        model, 
        (dummy_bids, dummy_asks), 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["bids", "asks"],
        output_names=["direction_logits", "spread_prob", "imbalance_pred"],
        dynamic_axes={
            "bids": {0: "batch_size"},
            "asks": {0: "batch_size"},
            "direction_logits": {0: "batch_size"},
            "spread_prob": {0: "batch_size"},
            "imbalance_pred": {0: "batch_size"},
        }
    )
    print(f"Dummy model successfully saved to {output_path}")

if __name__ == "__main__":
    generate_dummy_onnx()
