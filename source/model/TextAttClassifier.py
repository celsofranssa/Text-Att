import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam

from source.model.MaxPooling import MaxPooling
from source.model.MultiHeadAttention import MultiHeadAttention


class TextAttClassifier(LightningModule):
    """
    Attention text classification
    """

    def __init__(self, hparams):
        super(TextAttClassifier, self).__init__()
        self.hparams = hparams

        self.multihead_att = MultiHeadAttention(hparams)

        self.pool = MaxPooling()

        self.linear = nn.Linear(hparams.embed_dim, hparams.num_classes)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.loss = nn.NLLLoss()

    def forward(self, x):
        attn_output = self.multihead_att(x)
        pool_out = self.pool(attn_output)
        linear_out = self.linear(pool_out)
        return self.softmax(linear_out)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        train_loss = self.loss(y_hat, y)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)

    def validation_epoch_end(self, outs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outs):
        pass
