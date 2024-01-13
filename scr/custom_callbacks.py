#     def on_validation_epoch_end(self, trainer, pl_module):
#         # Access the logged metrics for the last validation epoch
#         metrics = trainer.callback_metrics
#         print(f"Metric form  valid: ", metrics)
#         val_loss = metrics.get('val_loss')
#         val_miss_penalty = metrics.get('val_miss_penalty')
#         if val_loss is not None and val_miss_penalty is not None:
#             print(
#                 f"Epoch {trainer.current_epoch}: Validation Loss: {val_loss}, Miss Penalty: {val_miss_penalty}")
from pytorch_lightning.callbacks import Callback, EarlyStopping

from scr.game import *


class LossLoggingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        total_loss = metrics.get('train_loss')
        miss_penalty = metrics.get('train_miss_penalty')
        if total_loss is not None and miss_penalty is not None:
            print(
                f"Epoch {trainer.current_epoch}: Training Total Loss: {total_loss}, Miss Penalty: {miss_penalty}")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # print(f"Metrics from validation: ", metrics)
        val_loss = metrics.get('val_loss_epoch')  # Assuming epoch-wise logging
        # Assuming epoch-wise logging
        val_miss_penalty = metrics.get('val_miss_penalty_epoch')
        win_rate = metrics.get('test_win_rate')  # Extract win rate

        if val_loss is not None and val_miss_penalty is not None:
            print(
                f"Epoch {trainer.current_epoch}: Validation Loss: {val_loss}, Miss Penalty: {val_miss_penalty}, Win Rate: {win_rate}")


# Setup EarlyStopping to monitor the test_win_rate
early_stop_callback = EarlyStopping(
    monitor='test_win_rate',
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='max'  # Maximize the win rate
)


# # EarlyStopping callback monitors 'val_loss' and stops training after 3 epochs if it doesn't improve
# early_stop_callback = EarlyStopping(
#     monitor='val_miss_penalty',  # Monitor the validation miss penalty
#     min_delta=0.00,  # Minimum change to qualify as an improvement
#     patience=3,  # Number of epochs with no improvement after which training will be stopped    ``
#     verbose=True,
#     mode='min'  # 'min' mode means training will stop when the quantity monitored has stopped decreasing
# )

# # EarlyStopping callback monitors 'test_win_rate' and stops training if it doesn't improve
# early_stop_callback = EarlyStopping(
#     monitor='test_win_rate',  # Monitor the test win rate
#     min_delta=0.00,
#     patience=3,
#     verbose=True,
#     mode='max'  # 'max' mode means training will stop when the quantity monitored has stopped increasing
# )


# class TestPerformanceCallback(Callback):
#     def __init__(self, test_words, char_frequency, max_word_length):
#         self.test_words = test_words
#         self.char_frequency = char_frequency
#         self.max_word_length = max_word_length

#     def on_sanity_check_end(self, trainer, pl_module):
#         self.evaluate_and_log(trainer, pl_module)

#     def on_validation_epoch_end(self, trainer, pl_module, outputs=None):
#         self.evaluate_and_log(trainer, pl_module)

#     def evaluate_and_log(self, trainer, pl_module):
#         pl_module.eval()
#         result = play_games_and_calculate_stats(pl_module, self.test_words,
#                                                 self.char_frequency, self.max_word_length)

#         current_win_rate = result['overall_win_rate']
#         trainer.logger.log_metrics(
#             {'test_win_rate': current_win_rate}, step=trainer.current_epoch)

#         print(f"\n===== Epoch {trainer.current_epoch} - Test Performance: =====")
#         print(
#             f"Overall Win Rate: {current_win_rate}%, Overall Average Attempts: {result['overall_avg_attempts']}")
#         for length, data in result["length_wise_stats"].items():
#             print(
#                 f"Length {length}: Win Rate: {data['win_rate']}%, Average Attempts: {data['average_attempts_used']}")

#         pl_module.train()


# class TestPerformanceCallback(Callback):
#     def __init__(self, test_words, char_frequency, max_word_length):
#         self.test_words = test_words
#         self.char_frequency = char_frequency
#         self.max_word_length = max_word_length

#     def on_train_epoch_end(self, trainer, pl_module, outputs=None):
#         # Ensure the model is in evaluation mode
#         pl_module.eval()

#         # Evaluate the model on the test words
#         result = play_games_and_calculate_stats(pl_module, self.test_words,
#                                                 self.char_frequency, self.max_word_length)

#         current_win_rate = result['overall_win_rate']
#         trainer.logger.log_metrics(
#             {'test_win_rate': current_win_rate}, step=trainer.current_epoch)

#         # Print the results
#         print(f"\nEpoch {trainer.current_epoch} - Test Performance:")
#         print(
#             f"Overall Win Rate: {result['overall_win_rate']}%, Overall Average Attempts: {result['overall_avg_attempts']}")
#         for length, data in result["length_wise_stats"].items():
#             print(
#                 f"Length {length}: Win Rate: {data['win_rate']}%, Average Attempts: {data['average_attempts_used']}")

#         # Set the model back to training mode
#         pl_module.train()


# # Initialize the TestPerformanceCallback
# test_performance_callback = TestPerformanceCallback(sampled_test_words, char_frequency, max_word_length)

# # Initialize a PyTorch Lightning Trainer - No need to set current_epoch
# trainer = pl.Trainer()  # Mock trainer object

# # Ensure the model is in the right state (e.g., evaluation mode)
# lightning_model.eval()

# # Call the method manually (passing a mock trainer and your model)
# # 'outputs' parameter can be None if not used
# test_performance_callback.on_train_epoch_end(trainer, lightning_model, outputs=None)

# # Set the model back to training mode if necessary
# lightning_model.train()
