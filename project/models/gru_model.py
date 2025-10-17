import torch
import torch.nn as nn
from .base_model import BaseModel

class GRUModel(BaseModel):
    """Enhanced GRU Model"""
    def __init__(self, num_features, num_classes, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer GRU
        self.gru = nn.GRU(
            num_features, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        # Reshape from (batch, features) to (batch, seq_len, features)
        x = x.unsqueeze(1)
        
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        
        # Take the last output
        last_output = gru_out[:, -1, :]
        
        # Classification layers
        output = self.dropout(last_output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output
