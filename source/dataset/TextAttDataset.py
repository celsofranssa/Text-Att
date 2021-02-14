import json

import torch
from torch.utils.data import Dataset


class TextAttDataset(Dataset):
    """Text-Att Dataset.
    """

    def __init__(self, path):
        super(TextAttDataset, self).__init__()
        self.samples = []
        self._init_dataset(path)

    def _init_dataset(self, dataset_path):
        with open(dataset_path, "r") as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                self.samples.append({
                    "id": sample["id"],
                    "x": torch.tensor(sample["x"]),
                    "y": sample["y"]
                })

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]
