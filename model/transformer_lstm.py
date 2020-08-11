from torch import nn
import copy
import torch
from torch.nn import functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, config,weights_ad):
        super(Model, self).__init__()

        self.embedding_ad = nn.Embedding.from_pretrained(weights_ad)

        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)位置编码层是否使用差别不大
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout,config.pad_size)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoder)])
        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)


    def forward(self, o):
        x ,x1 =o
        out = self.embedding_ad(x)
        for encoder in self.encoders:
            out = encoder(out ,x1)
        embed = out
        out, _ = self.lstm(out)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze(  )  # 128*(3*128)
        out= self.fc(out)

        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout,pad_size):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout,pad_size)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x ,x1=None):
        out = self.attention(x ,x1)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor \
            ([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None ,mask=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale

        if mask is not None:
            attention = attention *mask
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0, pad_size=100):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        self.pad_size = pad_size
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head  )  # 300,4*75
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x ,x1=None):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)

        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)

        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5  # 缩放因子
        if x1 is not None:
            mask = x1.repeat(1 ,self.num_head)
            mask = mask.view(batch_size *self.num_head ,1 ,self.pad_size)
            mask = mask.repeat(1 ,self.pad_size ,1)
            context = self.attention(Q, K, V, scale ,mask)
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out