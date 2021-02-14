import torch
from pytorch_lightning import LightningModule
from torch import nn

from source.model.MaxPooling import MaxPooling
from source.model.MultiHeadAttention import MultiHeadAttention


class TextAtt(LightningModule):
    """
    Attention text classification
    """

    def __init__(self, hparams):
        super(TextAtt, self).__init__()
        self.hparams = hparams

        self.multihead_att = MultiHeadAttention(hparams)

        self.pool = MaxPooling()

        self.linear = nn.Linear(hparams.embed_dim, hparams.num_classes)

        self.softmax = torch.nn.Softmax()

        self.loss = 1

    def forward(self, x, y):
        attn_output = self.multihead_attn(x)
        pool_out = self.pool(attn_output)
        linear_out = self.linear(pool_out)
        y_hat = self.softmax(linear_out)

        return self.loss_fn(y, y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True
        )

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        train_loss = self.loss_fn(y, y_hat)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        self.log("val_loss", self.loss_fn(y, y_hat), prog_bar=True)

    def validation_epoch_end(self, outs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outs):
        pass
