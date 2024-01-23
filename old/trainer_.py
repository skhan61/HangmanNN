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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from scr.feature_engineering import *
from scr.game import *
from scr.guess import guess as guess_fqn
from scr.utils import *


class HangmanModel(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate, char_frequency,
                 max_word_length, l1_factor=0.01, l2_factor=0.01,
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

        self.last_eval_metrics = {}

        # self.optimizer = torch.optim.Adam(
        #     self.parameters(), lr=self.learning_rate)
        # self.lr_schedulers = []  # Placeholder for scheduler(s)

    # def configure_optimizers(self):
    #     return [self.optimizer], self.lr_schedulers

    # def configure_optimizers(self):
    #     # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     # optimizer = torch.optim.RMSprop(
    #     #     self.parameters(), lr=self.learning_rate)

    #     # Estimate total steps
    #     # replace with actual dataset size
    #     dataset_size = len(self.trainer.datamodule.train_dataset)
    #     batch_size = self.trainer.datamodule.train_dataloader(
    #     ).batch_size  # replace with actual batch size
    #     total_steps = self.trainer.max_epochs * (dataset_size // batch_size)

    #     scheduler = {
    #         'scheduler': OneCycleLR(optimizer, max_lr=self.learning_rate,
    #                                 total_steps=total_steps),
    #         'interval': 'step'
    #     }

    #     return [optimizer], [scheduler]
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Retrieve the train dataloader and its sampler
        train_dataloader = self.trainer.datamodule.train_dataloader()
        sampler = train_dataloader.sampler

        # Use the sampler's length for total steps calculation
        dataset_size = len(sampler)
        batch_size = train_dataloader.batch_size
        total_steps = self.trainer.max_epochs * (dataset_size // batch_size)

        scheduler = {
            'scheduler': OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total_steps),
            'interval': 'step'
        }

        return [optimizer], [scheduler]

    def forward(self, fets, original_seq_lens, missed_chars):
        encoded_fets = self.encoder(fets, original_seq_lens, missed_chars)
        outputs = self.decoder(encoded_fets, original_seq_lens, missed_chars)
        return outputs

    def training_step(self, batch, batch_idx):
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        # # Print debug information
        # print("DEBUG INFO:")
        # print(f"Miss Penalty: {miss_penalty}")
        # print(f"Type of Miss Penalty: {type(miss_penalty)}")
        # print(f"Batch Word Lengths: {batch['word_length']}")
        # print(f"Batch Difficulties: {batch['difficulty']}")
        # print(f"Batch Outcomes: {batch['outcome']}")
        # print(f"Batch Won Flags: {batch['won']}")

        batch_features, batch_missed_chars = process_batch_of_games(
            states, self.char_frequency,
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

    # def on_train_epoch_end(self, unused=None):
    #     # print(f"from on_train_epoch_end: ", self.current_epoch)
    #     if self.current_epoch > 0:
    #         # Update to use custom sampler after first epoch
    #         self.trainer.datamodule.use_performance_based_sampling(True)

    def validation_step(self, batch, batch_idx):
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        # # Print debug information
        # print("DEBUG INFO:")
        # print(f"Miss Penalty: {miss_penalty}")
        # print(f"Type of Miss Penalty: {type(miss_penalty)}")
        # print(f"Batch Word Lengths: {batch['word_length']}")
        # print(f"Batch Difficulties: {batch['difficulty']}")
        # print(f"Batch Outcomes: {batch['outcome']}")
        # print(f"Batch Won Flags: {batch['won']}")

        # Process the batch
        batch_features, batch_missed_chars = process_batch_of_games(
            states, self.char_frequency, self.max_word_length, max_seq_length)

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

        # Initialize accumulator for granular statistics
        if not hasattr(self, 'granular_miss_penalty_stats'):
            self.granular_miss_penalty_stats = defaultdict(list)

        # Calculate miss penalty for each batch item
        batch_miss_penalties = []
        for i in range(outputs.size(0)):  # Iterate over batch dimension
            item_miss_penalty = self.calculate_miss_penalty(
                outputs[i], batch_missed_chars[i], original_seq_lengths_tensor[i])
            batch_miss_penalties.append(
                item_miss_penalty.item())  # Store scalar values

        # Segment the batch based on unique combinations
        unique_combinations = set(
            zip(batch['word_length'], batch['difficulty'], batch['outcome']))

        for combo in unique_combinations:
            wl, d, o = combo
            # Create a mask for the combination
            combo_mask = [(batch_len == wl and batch_diff == d and batch_out == o)
                          for batch_len, batch_diff, batch_out in
                          zip(batch['word_length'], batch['difficulty'], batch['outcome'])]

            # Filter and average miss penalties for this combination
            combo_miss_penalty_values = [batch_miss_penalties[i]
                                         for i, mask in enumerate(combo_mask) if mask]
            if combo_miss_penalty_values:  # Ensure non-empty list
                combo_avg_miss_penalty = sum(
                    combo_miss_penalty_values) / len(combo_miss_penalty_values)
                self.granular_miss_penalty_stats[combo].append(
                    combo_avg_miss_penalty)
            else:
                print(f"No data for combination {combo}, skipping.")

    def on_validation_epoch_end(self):
        # Evaluating and logging overall metrics
        eval_metrics = self.evaluate_and_log()
        self.last_eval_metrics = eval_metrics

        # Log existing metrics
        self.log('test_win_rate', eval_metrics['win_rate'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('overall_avg_attempts',
                 eval_metrics['attempts'], on_step=False, on_epoch=True, prog_bar=True)

        # Process and log granular miss penalty statistics
        granular_eval_metrics = {}
        for key, penalties in self.granular_miss_penalty_stats.items():
            avg_miss_penalty = sum(penalties) / len(penalties)
            metric_name = f'avg_miss_penalty_{key}'
            self.log(metric_name, avg_miss_penalty, on_epoch=True)
            granular_eval_metrics[metric_name] = avg_miss_penalty

        # Combine eval_metrics and granular_eval_metrics
        combined_eval_metrics = {**eval_metrics, **granular_eval_metrics}

        # print(type(combined_eval_metrics))

        # performance_metrics_dict = {
        #     k: v for d in combined_eval_metrics for k, v in d.items()}

        # # Updating the data module's sampler with new performance metrics
        # self.trainer.datamodule.update_performance_metrics(
        #     combined_eval_metrics)

        # # Print the combined evaluation metrics
        # print("Combined Evaluation Metrics:")
        # for key, value in combined_eval_metrics.items():
        #     print(f"{key}: {value}")

        # Clear granular statistics for the next epoch
        self.granular_miss_penalty_stats.clear()

        return combined_eval_metrics

    def evaluate_and_log(self):
        # Ensure the model is in evaluation mode
        self.eval()

        # Perform evaluation and gather statistics
        result = play_games_and_calculate_stats(
            self, self.test_words, self.char_frequency, self.max_word_length)

        # Extracting necessary statistics
        stats = result['stats']
        win_rate = result['overall_win_rate']
        attempts = result['overall_avg_attempts']
        length_wise_stats = result['length_wise_stats']

        # Switch back to training mode
        self.train()

        # Return the collected statistics
        return {'stats': stats, 'win_rate': win_rate,
                'attempts': attempts, 'length_wise_stats': length_wise_stats}

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

        # # Calculate weights for loss function
        # weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)

        # weights = weights_orig.unsqueeze(1).unsqueeze(
        #     2).expand(-1, outputs.size(1), -1)

        # Assign constant weights for debugging
        constant_weight_value = 1.0  # You can choose any constant value
        weights = torch.full(size=(outputs.size(0), outputs.size(1), labels.size(2)),
                             fill_value=constant_weight_value,
                             device=outputs.device)

        # print(f"weights shape: ", weights.shape)

        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')

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
        total_loss = loss + miss_penalty + l1_loss + l2_loss

        return total_loss, miss_penalty
