import pickle
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

    # def _encode(self, sample):
    #     return {
    #         "idx": sample["idx"],
    #         "rpr": torch.tensor(
    #             self.tokenizer.encode(text=sample["text"], max_length=self.max_length, padding="max_length",
    #                                   truncation=True)
    #         ),
    #         "cls": sample["cls"]
    #     }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_idx = self.ids[idx]
        return self._encode(
            self.samples[sample_idx]
        )

    # def __init__(self, path):
    #     super(TextAttDataset, self).__init__()
    #     self.samples = []
    #     self._init_dataset(path)

    # def _init_dataset(self, dataset_path):
    #     with open(dataset_path, "r") as dataset_file:
    #         for line in dataset_file:
    #             sample = json.loads(line)
    #             self.samples.append({
    #                 "id": sample["id"],
    #                 "x": torch.tensor(sample["x"]),
    #                 "y": sample["y"]
    #             })

    # def __len__(self):
    #     return len(self.samples)
    #
    #
    # def __getitem__(self, idx):
    #     return self.samples[idx]
