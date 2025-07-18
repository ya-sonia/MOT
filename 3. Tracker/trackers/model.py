import torch.nn as nn

class GRUPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
