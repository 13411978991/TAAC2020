from torch import nn
import torch
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, config,weights):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, o):
        x,x1=o
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out=self.fc(out)
        return out
