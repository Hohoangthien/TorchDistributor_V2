import torch
import torch.nn as nn
from .base_model import BaseModel

class SimpleRNNModel(BaseModel):
    """Basic RNN Model"""
    def __init__(self, num_features, num_classes, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            num_features, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Reshape from (batch, features) to (batch, seq_len, features)
        x = x.unsqueeze(1)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Take the last output
        last_output = rnn_out[:, -1, :]
        
        # Apply dropout and final classification layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output
