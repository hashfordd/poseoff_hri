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

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, C, V, M) padded batch
            mask: (B, T) bool, True for real timesteps and False for padding.
                None means every timestep is real, which is the correct
                behaviour for a single unpadded clip at inference time.
        """
        B, T, C, V, M = x.shape
        x = x.reshape(B, T, C * V *M)

        x = self.input(x)
        x = self.positional_encoding(x)

        # (B, T) -> (B, 1, 1, T) broadcasts over heads and query positions, so
        # padded keys are excluded from every query's attention distribution.
        attn_mask = None if mask is None else mask[:, None, None, :]

        for layer in self.encoder_layer:
            x = layer(x, mask=attn_mask)

        if mask is None:
            x = x.mean(dim=1)
        else:
            # Mean over real timesteps only. Plain .mean(dim=1) divides by the
            # padded length, so a short clip's signal is diluted in proportion to
            # whatever else happened to share its batch.
            m = mask.unsqueeze(-1).to(x.dtype)
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        return self.fc(x)
