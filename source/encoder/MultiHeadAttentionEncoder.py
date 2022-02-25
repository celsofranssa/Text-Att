import torch
from pytorch_lightning import LightningModule
from torch import nn


class MultiHeadAttentionEncoder(LightningModule):
    """
    Performs max pooling on the last hidden-states transformer output.
    """

    def __init__(self, hidden_size, num_heads, dropout, pooling):
        super(MultiHeadAttentionEncoder, self).__init__()

        self.key_fnn = nn.Linear(hidden_size, hidden_size)
        self.query_fnn = nn.Linear(hidden_size, hidden_size)
        self.value_fnn = nn.Linear(hidden_size, hidden_size)

        self.multihead_att = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pooling = pooling

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        key = self.key_fnn(x)
        query = self.query_fnn(x)
        value = self.value_fnn(x)
        attn_output, _ = self.multihead_att(query, key, value)
        return self.pooling(
            torch.transpose(attn_output, 0, 1)
        )


