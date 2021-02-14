import hydra
import os
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from source.DataModule.TextAttDataModule import TextAttDataModule
from source.model.TextAttClassifier import TextAttClassifier


def fit(settings):
    print("Fitting with the following parameters:\n", OmegaConf.to_yaml(settings))

    # TODO: implement logger

    # TODO: implement checkpoint callback

    # TODO: implement early stopping callback

    for fold in settings.folds:

        dm = TextAttDataModule(settings.data)

        model = TextAttClassifier(settings.model)

        trainer = Trainer(
            fast_dev_run=settings.trainer.fast_dev_run,
            max_epochs=settings.trainer.max_epochs,
            gpus=1
        )

        # training
        dm.setup('fit', fold)
        trainer.fit(model, datamodule=dm)


def predict(settings):
    print("Predicting with the following parameters:\n", OmegaConf.to_yaml(settings))

    # data
    dm = TextAttDataModule(settings.data)

    # model
    model = TextAttClassifier.load_from_checkpoint(
        checkpoint_path=settings.model_checkpoint.dir + settings.model.name + "_" + settings.data.name + ".ckpt"
    )

    # trainer
    trainer = Trainer(
        fast_dev_run=settings.trainer.fast_dev_run,
        max_epochs=settings.trainer.max_epochs,
        gpus=1
    )

    # testing
    dm.setup('test')
    trainer.test(model=model, datamodule=dm)


def eval(settings):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(settings))


def testing(settings):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(settings))
    dm = TextAttDataModule(settings.data)

    dm.setup('fit')

    #use data
    train_dataloader = dm.train_dataloader()

    for batch in train_dataloader:
        print(batch)
        break



@hydra.main(config_path="settings/", config_name="settings")
def perform_tasks(settings):
    os.chdir(hydra.utils.get_original_cwd())

    if "fit" in settings.tasks:
        fit(settings)
    if "predict" in settings.tasks:
        predict(settings)
    if "eval" in settings.tasks:
        eval(settings)
    if "test" in settings.tasks:
        testing(settings)


if __name__ == '__main__':
    perform_tasks()
