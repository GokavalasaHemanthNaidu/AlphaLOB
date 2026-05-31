import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class LOBTransformer(nn.Module):
    def __init__(self, n_levels: int = 10, features_per_level: int = 4, d_model: int = 64, n_heads: int = 8, n_layers: int = 6, dropout: float = 0.1):
        """
        LOBTransformer treats each price level (bid/ask pair) as a sequence token.
        Sequence length = n_levels.
        """
        super().__init__()
        self.n_levels = n_levels
        
        # Input projection: maps features_per_level to d_model
        self.input_proj = nn.Linear(features_per_level, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_levels)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # We will use the output for the multi-task head. 
        # Typically, we can flatten or pool the sequence output.
        # Here we just use flatten for simplicity, which yields a representation of size n_levels * d_model
        self.d_out = n_levels * d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, n_levels, features_per_level]
               representing the 10 levels of the Limit Order Book.
        Returns:
            Tensor of shape [batch_size, n_levels * d_model]
        """
        # [batch, seq_len, d_model]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # TransformerEncoder expects [batch, seq_len, d_model] if batch_first=True
        encoded = self.transformer_encoder(x)
        
        # Flatten the sequence
        # [batch_size, n_levels * d_model]
        return encoded.flatten(start_dim=1)
