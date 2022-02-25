import hydra
import os
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.FitHelper import FitHelper


def fit(params):
    fit_helper = FitHelper(params)
    fit_helper.perform_fit()


def eval(params):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(params))
    eval_helper = EvalHelper(params)
    eval_helper.perform_eval()


@hydra.main(config_path="settings", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "fit" in params.tasks:
        fit(params)
    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()
