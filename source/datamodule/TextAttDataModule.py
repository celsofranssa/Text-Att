import pickle

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

        # Assign train/val dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = TextAttDataset(
                samples=self.samples,
                ids_path=f"{self.params.dir}fold_{self.params.fold_id}/train.pkl"
            )

            self.val_dataset = TextAttDataset(
                samples=self.samples,
                ids_path=f"{self.params.dir}fold_{self.params.fold_id}/val.pkl"
            )

        if stage == 'test' or stage == "predict":
            self.test_dataset = TextAttDataset(
                samples=self.samples,
                ids_path=f"{self.params.dir}fold_{self.params.fold_id}/test.pkl"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()
