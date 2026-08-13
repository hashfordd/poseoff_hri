import torch.nn as nn
from Training.encoder import EncoderLayer, PositionalEncoding

class transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, input_dim, num_classes):
        super(transformer, self).__init__()
        self.input = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length = 5000)

        self.encoder_layer = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C, V, M = x.shape
        x = x.reshape(B, T, C * V *M)

        x = self.input(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layer:
            x = layer(x, mask=None)

        x = x.mean(dim=1)

        return self.fc(x) 
    

        
