import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import F1
from torch import nn
from torch.optim import Adam
from torchmetrics import MetricCollection

from source.pooling.MaxPooling import MaxPooling
from torch import nn


class TextAttModel(LightningModule):
    """
    Attention text classification
    """

    def __init__(self, hparams):
        super(TextAttModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.att_encoder = self.encoder = instantiate(hparams.att_encoder)

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.hidden_size, hparams.num_classes),
            torch.nn.LogSoftmax(dim=-1)
        )

        self.loss = nn.CrossEntropyLoss()
        self.val_metrics = self._get_metrics(prefix="val_")
        self.test_metrics = self._get_metrics(prefix="test_")

    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "Mic-F1": F1(num_classes=self.hparams.num_classes, average="micro"),
                "Mac-F1": F1(num_classes=self.hparams.num_classes, average="macro"),
                "Wei-F1": F1(num_classes=self.hparams.num_classes, average="weighted")
            },
            prefix=prefix)

    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True)
        return optimizer

    def forward(self, rpr):
        return self.cls_head(
            self.att_encoder(rpr)
        )

    def training_step(self, batch, batch_idx):
        rpr, true_cls = batch["rpr"], batch["cls"]
        pred_cls = self(rpr)
        train_loss = self.loss(pred_cls, true_cls)
        train_loss = self.loss(pred_cls, true_cls)

        # log training loss
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        rpr, true_cls = batch["rpr"], batch["cls"]
        pred_cls = self(rpr)
        val_loss = self.loss(pred_cls, true_cls)
        # log val loss
        self.log('val_loss', val_loss)

        # log val metrics
        self.log_dict(self.val_metrics(pred_cls, true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def validation_epoch_end(self, outs):
        pass

    def test_step(self, batch, batch_idx):
        rpr, true_cls = batch["rpr"], batch["cls"]
        pred_cls = self(rpr)

        # log test metrics
        self.log_dict(self.test_metrics(pred_cls, true_cls), prog_bar=True)

    def test_epoch_end(self, outs):
        test_result = self.test_metrics.compute()
        print(test_result)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, rpr, true_cls = batch["idx"], batch["rpr"], batch["cls"]

        return {
                "idx": idx,
                "true_cls": true_cls,
                "pred_cls": torch.argmax(self(rpr), dim=-1)
            }
