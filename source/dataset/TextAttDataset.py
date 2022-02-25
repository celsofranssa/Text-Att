import pickle

import torch
from torch.utils.data import Dataset


class TextAttDataset(Dataset):
    """Text-Att Dataset.
    """

    def __init__(self, samples, ids_path):
        super(TextAttDataset, self).__init__()
        self.samples = samples
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def __len__(self):
        return len(self.ids)

    def _encode(self, sample):
        return {
            "idx": sample["idx"],
            "rpr": torch.tensor(sample["rpr"]),
            "cls": sample["cls"]
        }

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        return self._encode(
            self.samples[sample_id]
        )
