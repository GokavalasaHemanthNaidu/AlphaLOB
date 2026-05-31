import torch
import torch.nn as nn
from typing import Tuple

class MultiTaskHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared representations can go here if needed, but we'll use separate heads for clarity
        
        # 1. Directional Heads (Classification: UP prob) for 3 horizons: 5s, 30s, 5min
        # We output logits; Sigmoid will be applied during inference or BCEWithLogitsLoss during training
        self.dir_5s_head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dir_30s_head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dir_5m_head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 2. Spread Compression Head (Binary classification: will spread decrease?)
        self.spread_compress_head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 3. Volume Imbalance Head (Regression: continuous output)
        self.vol_imbalance_head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 4. Kendall Uncertainty Weighting (Learnable log variances for 5 tasks)
        # s_k = log(sigma_k^2). Initialized to 0.
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor, shape [batch_size, d_in]
               Usually the flattened output from LOBTransformer.
               
        Returns:
            Tuple of 5 Tensors:
            - dir_5s (logits)
            - dir_30s (logits)
            - dir_5min (logits)
            - spread_compress (logits)
            - vol_imbalance (regression value)
        """
        # We apply sigmoid here so that the ONNX export bakes the sigmoid into the graph
        # This means the FastAPI inference code gets raw probabilities [0, 1] directly.
        dir_5s = torch.sigmoid(self.dir_5s_head(x))
        dir_30s = torch.sigmoid(self.dir_30s_head(x))
        dir_5m = torch.sigmoid(self.dir_5m_head(x))
        spread_compress = torch.sigmoid(self.spread_compress_head(x))
        
        # Volume imbalance is regression, no sigmoid
        vol_imbalance = self.vol_imbalance_head(x)
        
        return dir_5s, dir_30s, dir_5m, spread_compress, vol_imbalance

    def compute_loss(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Computes the Kendall et al. (2018) uncertainty-weighted multi-task loss.
        L = sum_k ( exp(-s_k) * L_k + 0.5 * s_k )
        """
        assert len(losses) == 5, "Expected 5 task losses"
        
        total_loss = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + 0.5 * self.log_vars[i]
            
        return total_loss

class AlphaLOBModel(nn.Module):
    def __init__(self, n_levels: int = 10, features_per_level: int = 4, d_model: int = 64):
        super().__init__()
        # Avoid circular imports in this mock environment, usually we'd import LOBTransformer here
        from src.domain.models.lob_transformer import LOBTransformer
        self.transformer = LOBTransformer(n_levels=n_levels, features_per_level=features_per_level, d_model=d_model)
        self.multi_task = MultiTaskHead(d_in=self.transformer.d_out)
        
    def forward(self, x: torch.Tensor):
        features = self.transformer(x)
        return self.multi_task(features)
