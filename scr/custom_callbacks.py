from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR

from scr.game import *

# Setup EarlyStopping to monitor the test_win_rate
early_stop_callback = EarlyStopping(
    monitor='test_win_rate',
    min_delta=0.00,
    patience=20,
    verbose=True,
    mode='max'  # Maximize the win rate
)


# class SchedulerSetupCallback(Callback):
#     def on_train_start(self, trainer, pl_module):
#         num_epochs = trainer.max_epochs
#         num_training_batches = len(trainer.train_dataloader)

#         total_steps = num_epochs * num_training_batches
#         total_steps = max(1, total_steps)  # Ensure it's at least 1

#         max_lr = 0.01  # Set your max_lr
#         scheduler = OneCycleLR(pl_module.optimizer, max_lr=max_lr,
#                                total_steps=total_steps, anneal_strategy='linear')

#         pl_module.lr_schedulers = [scheduler]
