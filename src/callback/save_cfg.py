import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from src.config import read_config, save_config

class SaveCfg(pl.Callback):
    def __init__(self, cfg=None, wandb_path=None):
        super().__init__()
        self._attributes = {}
        self.cfg = cfg
        self.wandb_path = wandb_path

    def __setitem__(self, key, value):
        self._attributes[key] = value

    def __getitem__(self, key):
        return self._attributes[key]

    def on_train_epoch_end(self, trainer, pl_module):
        print('Inside on_train_epoch_end')
        for key, value in self._attributes.items():
            print(f'{key}: {value}')