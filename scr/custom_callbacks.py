from pytorch_lightning.callbacks import Callback, EarlyStopping

from scr.game import *


# class LossLoggingCallback(Callback):
#     def on_train_epoch_end(self, trainer, pl_module):
#         metrics = trainer.callback_metrics
#         total_loss = metrics.get('train_loss')
#         miss_penalty = metrics.get('train_miss_penalty')
#         if total_loss is not None and miss_penalty is not None:
#             print(
#                 f"Epoch {trainer.current_epoch}: Training Total Loss: {total_loss}, Miss Penalty: {miss_penalty}")

#     def on_validation_epoch_end(self, trainer, pl_module):
#         metrics = trainer.callback_metrics
#         # print(f"Metrics from validation: ", metrics)
#         val_loss = metrics.get('val_loss_epoch')  # Assuming epoch-wise logging
#         # Assuming epoch-wise logging
#         val_miss_penalty = metrics.get('val_miss_penalty_epoch')
#         win_rate = metrics.get('test_win_rate')  # Extract win rate

#         if val_loss is not None and val_miss_penalty is not None:
#             print(
#                 f"Epoch {trainer.current_epoch}: Validation Loss: {val_loss}, Miss Penalty: {val_miss_penalty}, Win Rate: {win_rate}")


# Setup EarlyStopping to monitor the test_win_rate
early_stop_callback = EarlyStopping(
    monitor='test_win_rate',
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='max'  # Maximize the win rate
)
