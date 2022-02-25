from omegaconf import OmegaConf

from source.callback.PredictionWriter import PredictionWriter
from source.datamodule.TextAttDataModule import TextAttDataModule
from source.model.TextAttModel import TextAttModel

import pytorch_lightning as pl


class PredictHelper:

    def __init__(self, params):
        self.params = params

    def perform_predict(self):
        for fold_id in self.params.data.folds:
            print(f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling params\n"
                  f"{OmegaConf.to_yaml(self.params)}\n")

            # data
            self.params.data.fold_id = fold_id
            self.params.prediction.fold_id = fold_id
            dm = TextAttDataModule(self.params.data)
            dm.prepare_data()
            dm.setup("predict")

            # model
            model = TextAttModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_id}.ckpt"
            )



            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[PredictionWriter(self.params.prediction)]
            )

            trainer.predict(
                model=model,
                datamodule=dm,

            )
