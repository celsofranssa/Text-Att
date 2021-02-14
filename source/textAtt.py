import os
import hydra
from omegaconf import OmegaConf


def fit(settings):
    print("Fitting with the following parameters:\n", OmegaConf.to_yaml(settings))


def predict(settings):
    print("Predicting with the following parameters:\n", OmegaConf.to_yaml(settings))


def eval(settings):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(settings))


@hydra.main(config_path="settings/", config_name="settings")
def perform_tasks(settings):
    os.chdir(hydra.utils.get_original_cwd())

    if "fit" in settings.tasks:
        fit(settings)
    if "predict" in settings.tasks:
        predict(settings)
    if "eval" in settings.tasks:
        eval(settings)


if __name__ == '__main__':
    perform_tasks()
