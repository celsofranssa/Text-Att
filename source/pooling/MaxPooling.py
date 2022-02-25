import torch
from pytorch_lightning import LightningModule


class MaxPooling(LightningModule):
    """
    Performs max pooling on the last hidden-states attention encoder output.
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, hidden_states):
        return torch.max(hidden_states, 1)[0]


