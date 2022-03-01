import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


class EvalHelper:
    def __init__(self, params):
        self.params = params

    def checkpoint_stats(self, stats):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        stats.to_csv(
            self.params.stat.dir + self.params.model.name + "_" + self.params.data.name + ".stat",
            sep='\t', index=False, header=True)

    def load_predictions(self, fold):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold}/").glob("*.prd")
        )

        predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            predictions.extend(torch.load(path))

        return predictions

    def perform_eval(self):
        stats = pd.DataFrame(columns=["fold"])

        for fold in self.params.data.folds:
            true_classes = []
            pred_classes = []
            predictions = self.load_predictions(fold)
            for prediction in predictions:
                true_classes.append(prediction["true_cls"])
                pred_classes.append(prediction["pred_cls"])
            stats.at[fold, "Mic-F1"] = f1_score(true_classes, pred_classes, average='micro')
            stats.at[fold, "Mac-F1"] = f1_score(true_classes, pred_classes, average='macro')
            stats.at[fold, "Wei-F1"] = f1_score(true_classes, pred_classes, average='weighted')

        # update fold colum
        stats["fold"] = stats.index

        self.checkpoint_stats(stats)
