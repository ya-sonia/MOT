import torch
import torch.nn as nn

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=4, model_dim=64, num_heads=4, num_layers=2, output_dim=4, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.input_proj(x)         # [B, T, model_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # return self.fc_out(x[:, -1, :])  # predict from last time step
        return self.fc_out(x.mean(dim=1)) # aggregate over time steps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
