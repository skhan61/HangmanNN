from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import SimpleLSTM
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from scr.feature_engineering import *
from scr.game import *
from scr.guess import guess as guess_fqn
from scr.utils import *


class HangmanModel(pl.LightningModule):
    def __init__(self,
                 encoder,
                 decoder,
                 learning_rate,
                 char_frequency,
                 max_word_length,
                 optimizer_type='Adam',
                 l1_factor=0.01,
                 l2_factor=0.01,
                 test_words=None):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.learning_rate = learning_rate
        self.char_frequency = char_frequency
        self.max_word_length = max_word_length

        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.test_words = test_words

        # Predicted and actual guesses for metrics calculation
        self.predicted_guesses = []
        self.actual_guesses = []

        # Last evaluated metrics
        self.last_eval_metrics = {}

        # Initialize dictionaries for validation metrics
        self.validation_epoch_metrics = defaultdict(float)
        self.granular_miss_penalty_stats = defaultdict(list)

        self.optimizer_type = optimizer_type  # New parameter for optimizer type

    def configure_optimizers(self):
        if self.optimizer_type == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'AdamW':
            optimizer = SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'RMSprop':
            optimizer = RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.optimizer_type}")

        # Calculate steps per epoch
        dataset_size = len(self.trainer.datamodule.train_dataloader().dataset)
        batch_size = self.trainer.datamodule.train_dataloader().batch_size
        steps_per_epoch = max(dataset_size // batch_size,
                              1)  # Avoid division by zero

        # Calculate total steps
        total_steps = self.trainer.max_epochs * steps_per_epoch
        if total_steps <= 0:
            raise ValueError("Total steps must be a positive integer.")

        scheduler = {
            'scheduler': OneCycleLR(optimizer,
                                    max_lr=self.learning_rate,
                                    total_steps=total_steps),
            'interval': 'step'
        }

        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.learning_rate)

    #     scheduler = ReduceLROnPlateau(
    #         optimizer, mode='min', factor=0.01, patience=3, verbose=True)

    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': scheduler,
    #             'monitor': 'val_miss_penalty', # 'miss_penalty_step',  # Metric to monitor
    #             'interval': 'step',  # The scheduler will check the metric every epoch
    #             'frequency': 1,  # How frequently to check the metric
    #             'reduce_on_plateau': True  # This is specific to ReduceLROnPlateau
    #         }
    #     }

    def validation_step(self, batch, batch_idx):
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        # print(f"{batch}")

        # # Print debug information
        # print("DEBUG INFO:")
        # print(f"Miss Penalty: {miss_penalty}")
        # print(f"Type of Miss Penalty: {type(miss_penalty)}")
        # print(f"Batch Word Lengths: {batch['word_length']}")
        # print(f"Batch Difficulties: {batch['difficulty']}")
        # print(f"Batch Outcomes: {batch['outcome']}")
        # print(f"Batch Won Flags: {batch['won']}")

        batch_features, batch_missed_chars = process_batch_of_games(
            states, guesses, self.char_frequency,
            self.max_word_length,
            max_seq_length)

        original_seq_lengths_tensor = torch.tensor(original_seq_lengths,
                                                   dtype=torch.long,
                                                   device=self.device)

        encoded_guess = pad_and_reshape_labels(guesses, max_seq_length)
        encoded_guess = encoded_guess.to(self.device)  # Move to correct device

        # Move batch_missed_chars to the correct device
        batch_features = batch_features.to(self.device)
        batch_missed_chars = batch_missed_chars.to(self.device)

        # Forward pass
        outputs = self(
            batch_features, original_seq_lengths_tensor, batch_missed_chars)

        # Calculate loss and miss_penalty
        loss, miss_penalty = self.calculate_loss(
            outputs, encoded_guess, original_seq_lengths_tensor, batch_missed_chars)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=len(states))
        self.log('val_miss_penalty', miss_penalty, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=len(states))

        # Update the validation_epoch_metrics dictionary
        self.validation_epoch_metrics['validation_epoch_loss_sum'] += loss.item()
        self.validation_epoch_metrics['validation_epoch_miss_penalty_sum'] \
            += miss_penalty.item()

        # Initialize accumulator for granular statistics based on sequence length
        if not hasattr(self, 'granular_seq_len_miss_penalty_stats'):
            self.granular_seq_len_miss_penalty_stats = defaultdict(list)

        # Calculate miss penalty for each batch item
        for i in range(outputs.size(0)):
            item_miss_penalty = self.calculate_miss_penalty(
                outputs[i], batch_missed_chars[i], original_seq_lengths_tensor[i])
            seq_len = original_seq_lengths[i]
            self.granular_seq_len_miss_penalty_stats[seq_len].append(
                item_miss_penalty.item())

        # Initialize accumulator for granular statistics based on word length
        if not hasattr(self, 'granular_miss_penalty_stats'):
            self.granular_miss_penalty_stats = defaultdict(list)

        # Calculate miss penalty for each batch item
        batch_miss_penalties = []
        for i in range(outputs.size(0)):
            item_miss_penalty = self.calculate_miss_penalty(
                outputs[i], batch_missed_chars[i], original_seq_lengths_tensor[i])
            batch_miss_penalties.append(
                (batch['word_length'][i], item_miss_penalty.item()))

        # Group by word length
        unique_word_lengths = set(batch['word_length'])
        for length in unique_word_lengths:
            penalties_for_length = [
                penalty for word_len, penalty in batch_miss_penalties if word_len == length]
            if penalties_for_length:
                avg_penalty_for_length = sum(
                    penalties_for_length) / len(penalties_for_length)
                self.granular_miss_penalty_stats[length].append(
                    avg_penalty_for_length)
            else:
                print(f"No data for word length {length}, skipping.")

    def on_validation_epoch_end(self):
        # Evaluating and logging overall metrics from the custom evaluation method
        eval_metrics = self.evaluate_and_log()
        self.last_eval_metrics = eval_metrics

        # Initialize an empty dictionary to hold all performance stats
        performance_stats = {}

        # Log existing metrics
        self.log('win_rate', eval_metrics.get('win_rate', 0),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_attempts', eval_metrics.get('attempts', 0),
                 on_step=False, on_epoch=True, prog_bar=True)

        # Extract sequence length-wise win rates from simplified summaries
        simplified_summaries = eval_metrics['simplified_summaries']
        seq_len_win_rates = calculate_mean_win_rate_by_seq_length(
            simplified_summaries)

        # Prepare for aggregating sequence length-wise stats
        seq_length_wise_stats = {}
        for seq_length, penalties in self.granular_seq_len_miss_penalty_stats.items():
            if penalties:
                avg_penalty = sum(penalties) / len(penalties)
                seq_length_wise_stats[f'miss_penalty_seq_len_{seq_length}'] = avg_penalty

        # Add sequence length-wise win rates to seq_length_wise_stats
        for seq_len, win_rate in seq_len_win_rates.items():
            seq_length_wise_stats[f'win_rate_seq_len_{seq_len}'] = win_rate

        # # Debug: Print the seq_length_wise_stats dictionary
        # print("Seq Length Wise Stats:", seq_length_wise_stats)

        # Process seq_length_wise_stats and add them to performance_stats
        for seq_length, metrics in seq_length_wise_stats.items():
            if isinstance(metrics, dict):  # Check if metrics is a dictionary
                for metric_name, value in metrics.items():
                    performance_stats[f'seq_length_{seq_length}_{metric_name}'] = value
            else:  # Handle the case where metrics is a float
                # Assuming the float value represents a metric
                performance_stats[f'seq_length_{seq_length}'] = metrics

        # Log all performance stats
        for key, value in performance_stats.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False)

        epoch = self.trainer.current_epoch  # Get the current epoch from the Trainer
        base_dir = "/home/sayem/Desktop/Hangman"
        plot_and_save_win_rates(seq_len_win_rates, base_dir, epoch)

        # Additional steps...

        composite_scores = calculate_composite_scores(performance_stats)
        # print(composite_scores)

        # # Update the data module's sampler with new performance metrics, if applicable
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule:
            self.trainer.datamodule.update_performance_metrics(
                composite_scores)

        return performance_stats

    def evaluate_and_log(self):
        # Ensure the model is in evaluation mode
        self.eval()

        # Perform evaluation and gather statistics
        result = play_games_and_calculate_stats(
            self, self.test_words, self.char_frequency, self.max_word_length)

        # Extracting necessary statistics
        word_stats = result['stats']
        win_rate = result['overall_win_rate']
        attempts = result['overall_avg_attempts']
        length_wise_stats = result['length_wise_stats']
        # Extract simplified summaries
        simplified_summaries = result['simplified_summaries']

        self.train()  # Ensure the model is set back to training mode

        # Return the collected statistics, including simplified_summaries
        return {
            'win_rate': win_rate,
            'attempts': attempts,
            'length_wise_stats': length_wise_stats,
            # Include simplified summaries in the returned dictionary
            'simplified_summaries': simplified_summaries
        }

    def forward(self, fets, original_seq_lens, missed_chars):
        encoded_fets = self.encoder(fets, original_seq_lens, missed_chars)
        # print(f"{encoded_fets.shape}")
        outputs = self.decoder(encoded_fets, original_seq_lens, missed_chars)
        return outputs

    def training_step(self, batch, batch_idx):
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        # print(f"{batch}")

        # # Print debug information
        # print("DEBUG INFO:")
        # print(f"Miss Penalty: {miss_penalty}")
        # print(f"Type of Miss Penalty: {type(miss_penalty)}")
        # print(f"Batch Word Lengths: {batch['word_length']}")
        # print(f"Batch Difficulties: {batch['difficulty']}")
        # print(f"Batch Outcomes: {batch['outcome']}")
        # print(f"Batch Won Flags: {batch['won']}")

        batch_features, batch_missed_chars = process_batch_of_games(
            states, guesses, self.char_frequency,
            self.max_word_length,
            max_seq_length)

        original_seq_lengths_tensor = torch.tensor(original_seq_lengths,
                                                   dtype=torch.long,
                                                   device=self.device)

        encoded_guess = pad_and_reshape_labels(guesses, max_seq_length)
        encoded_guess = encoded_guess.to(self.device)  # Move to correct device

        # Move batch_missed_chars to the correct device
        batch_features = batch_features.to(self.device)
        batch_missed_chars = batch_missed_chars.to(self.device)

        # encoder_batch_fets = self.encoder(batch_features)  # feature encoder

        outputs = self(
            batch_features, original_seq_lengths_tensor, batch_missed_chars)

        # loss, miss_penalty = self.calculate_loss(outputs,
        #     encoded_guess, original_seq_lengths_tensor, batch_missed_chars)

        # Calculate loss with regularization
        loss, miss_penalty = self.calculate_loss(outputs, encoded_guess,
                                                 original_seq_lengths_tensor,
                                                 batch_missed_chars,
                                                 self.l1_factor,
                                                 self.l2_factor)

        # If you know the batch size, explicitly provide it when logging
        # Or however you can determine the batch size
        batch_size = len(batch['guessed_states'])

        self.log('total_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)

        self.log('miss_penalty', miss_penalty, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)

        return loss

    def calculate_miss_penalty(self, outputs, miss_chars, input_lens):
        # Ensure no empty tensors
        if outputs.numel() == 0:
            print("Empty outputs tensor")
            return torch.tensor(0.0, device=outputs.device)

        miss_penalty = torch.sum(outputs * miss_chars) / outputs.numel()
        return miss_penalty

    def calculate_loss(self, outputs, labels, input_lens,
                       miss_chars, l1_factor=0.01, l2_factor=0.01):

        # # print(f"model out: ", model_out.shape)
        # # outputs = torch.sigmoid(model_out)
        # outputs = model_out

        # print(f"print output shape: ", outputs.shape)
        # print(f"label shape: ", labels.shape)
        # print(f"input_lens: ", input_lens)
        # print(f"miss_chars: ", miss_chars.shape)

        # miss_penalty = torch.sum(outputs * miss_chars) / outputs.numel()
        miss_penalty = self.calculate_miss_penalty(
            outputs, miss_chars, input_lens)
        # print(f"miss_penalty: ", miss_penalty)

        # Calculate weights for loss function
        weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)

        weights = weights_orig.unsqueeze(1).unsqueeze(
            2).expand(-1, outputs.size(1), -1)

        # # Assign constant weights for debugging
        # constant_weight_value = 1.0  # You can choose any constant value
        # weights = torch.full(size=(outputs.size(0), outputs.size(1), labels.size(2)),
        #                      fill_value=constant_weight_value,
        #                      device=outputs.device)

        # print(f"weights shape: ", weights.shape)

        loss_func = nn.BCEWithLogitsLoss(weight=weights,
                                         reduction='none')

        # print()
        # print(f'{labels}')
        # print()

        # 'outputs' are logits, 'labels' are 0/1
        loss = loss_func(outputs, labels)

        # Ensure seq_lens_mask is on the same device as model_out
        seq_lens_mask = torch.arange(outputs.size(1), device=outputs.device).expand(
            len(input_lens), outputs.size(1)) < input_lens.unsqueeze(1)
        # print(f"seq_lens_mask shape: ", seq_lens_mask.shape)

        loss = loss * seq_lens_mask.unsqueeze(-1).float()
        loss = loss.sum() / seq_lens_mask.sum()

        # print(f"final loss: ", loss)

        # L1 Regularization
        l1_reg = torch.tensor(0., requires_grad=True, device=self.device)

        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)  # Non-in-place addition
        l1_loss = l1_factor * l1_reg

        # L2 Regularization
        l2_reg = torch.tensor(0., requires_grad=True, device=self.device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)  # Non-in-place addition
        l2_loss = l2_factor * l2_reg

        # Final loss
        total_loss = loss + 0.5 * miss_penalty + l1_loss + l2_loss

        return total_loss, miss_penalty
