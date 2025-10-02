import torch
import torch.nn as nn
from .base_model import BaseModel

class TransformerModel(BaseModel):
    """Transformer Model for sequence classification"""
    def __init__(self, num_features, num_classes, hidden_size=128, num_heads=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        
        # Input projection
        self.input_projection = nn.Linear(num_features, hidden_size)
        
        # Positional encoding (simple learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape from (batch, features) to (batch, seq_len, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        seq_len = x.size(1)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden_size)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch, hidden_size)
        
        # Classification
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
