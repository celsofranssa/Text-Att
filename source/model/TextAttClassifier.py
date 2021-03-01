import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import F1
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

        self.f1_score = F1(num_classes=self.hparams.num_classes, average='macro')

    def forward(self, x):
        attn_output = self.multihead_att(x)
        pool_out = self.pool(attn_output)
        linear_out = self.linear(pool_out)
        return self.softmax(linear_out), pool_out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat, _ = self(x)
        train_loss = self.loss(y_hat, y)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat, _ = self(x)
        val_loss = self.loss(y_hat, y)
        f1 = self.f1_score(y, torch.argmax(y_hat, dim=1))
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("F1", f1, prog_bar=True)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('F1', self.f1_score.compute())

    def test_step(self, batch, batch_idx):
        id, x, y = batch["id"], batch["x"], batch["y"]
        y_hat, text_representation = self(x)
        self.write_prediction_dict({
            "id": id,
            "text_representation": text_representation
        }, self.hparams.predictions.path)
        self.log('test_f1', self.f1_score(y, torch.argmax(y_hat, dim=1)), prog_bar=True)

    def test_epoch_end(self, outs):
        self.log('m_test_mrr', self.f1_score.compute())
