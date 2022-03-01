import torch
from pytorch_lightning import LightningModule


class AVGPooling(LightningModule):
    """
    Performs avg pooling on the last hidden-states attention encoder output.
    """

    def __init__(self):
        super(AVGPooling, self).__init__()

    def forward(self, hidden_states):
        raise NotImplemented


