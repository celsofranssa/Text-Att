from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class PredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(PredictionWriter, self).__init__(params.write_interval)
        self.params = params

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):
        predictions = []

        for idx, true_cls, pred_cls in zip(
                prediction["idx"].tolist(),
                prediction["true_cls"].tolist(),
                prediction["pred_cls"].tolist()
        ):
            predictions.append({
                "idx": idx,
                "true_cls": true_cls,
                "pred_cls": pred_cls
            })

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        dir = f"{self.params.dir}fold_{self.params.fold_id}/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            predictions,
            f"{dir}{dataloader_idx}_{batch_idx}.prd"
        )
