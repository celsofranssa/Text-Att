import torch
from pytorch_lightning import LightningModule
from torch import nn


class MultiHeadAttention(LightningModule):
    """
    Performs max pooling on the last hidden-states transformer output.
    """

    def __init__(self, hparams):
        super(MultiHeadAttention, self).__init__()

        self.hparams = hparams
        self.key_fnn = nn.Linear(hparams.embed_dim, hparams.embed_dim)
        self.query_fnn = nn.Linear(hparams.embed_dim, hparams.embed_dim)
        self.value_fnn = nn.Linear(hparams.embed_dim, hparams.embed_dim)

        self.multihead_att = torch.nn.MultiheadAttention(
            embed_dim=hparams.embed_dim,
            num_heads=hparams.num_heads,
            dropout=hparams.dropout
        )

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        key = self.key_fnn(x)
        query = self.query_fnn(x)
        value = self.value_fnn(x)
        attn_output, _ = self.multihead_att(query, key, value)
        return torch.transpose(attn_output, 0, 1)


