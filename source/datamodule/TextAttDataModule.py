import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.TextAttDataset import TextAttDataset


class TextAttDataModule(pl.LightningDataModule):
    """
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def prepare_data(self):
        samples_path = f"{self.params.dir}" \
                       f"fold_{self.params.fold_id}/" \
                       f"samples.pkl"
        with open(samples_path, "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

    def setup(self, stage=None):

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