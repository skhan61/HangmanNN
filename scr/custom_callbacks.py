from pytorch_lightning import Callback
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR

from scr.game import *


class StepLevelEarlyStopping(Callback):
    def __init__(self, monitor='val_miss_penalty', min_delta=0.0, patience=0):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait_count = 0
        self.best_score = None
        self.stop_training = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        current_score = trainer.logged_metrics.get(self.monitor)

        if current_score is not None:
            if self.best_score is None:
                self.best_score = current_score
            elif current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.wait_count = 0
            else:
                self.wait_count += 1

            if self.wait_count >= self.patience:
                self.stop_training = True

        if self.stop_training:
            trainer.should_stop = True
            print(
                f"Stopping training at step {batch_idx} of epoch {trainer.current_epoch} due to no improvement in {self.monitor}")

    def on_train_epoch_start(self, trainer, pl_module):
        self.wait_count = 0
        self.stop_training = False
