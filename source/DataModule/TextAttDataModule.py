import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.TextAttDataset import TextAttDataset


class TextAttDataModule(pl.LightningDataModule):
    """
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def prepare_data(self):
        pass

    def setup(self, stage=None, fold=0):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = TextAttDataset(
                path=f"{self.hparams.dir}/fold_{fold}/train.jsonl"
            )

            self.val_dataset = TextAttDataset(
                path=f"{self.hparams.dir}/fold_{fold}/test.jsonl"
            )

        if stage == 'test':
            self.test_dataset = TextAttDataset(
                path=f"{self.hparams.dir}/fold_{fold}/test.jsonl"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )