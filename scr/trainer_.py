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

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.RMSprop(
        #     self.parameters(), lr=self.learning_rate)

        # Estimate total steps
        # replace with actual dataset size
        dataset_size = len(self.trainer.datamodule.train_dataset)
        batch_size = self.trainer.datamodule.train_dataloader(
        ).batch_size  # replace with actual batch size
        total_steps = self.trainer.max_epochs * (dataset_size // batch_size)

        scheduler = {
            'scheduler': OneCycleLR(optimizer, max_lr=self.learning_rate,
                                    total_steps=total_steps),
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

        # Calculate loss (assuming your loss function is compatible with the format of your outputs and labels)
        loss, miss_penalty = self.calculate_loss(
            outputs, encoded_guess, original_seq_lengths_tensor, batch_missed_chars)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=len(states))
        self.log('val_miss_penalty', miss_penalty, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=len(states))

        # self.log('val_f1_score', f1, on_step=True, on_epoch=True,
        #          prog_bar=True, batch_size=len(states))

    def on_validation_epoch_end(self):
        eval_metrics = self.evaluate_and_log()

        # Store the metrics
        self.last_eval_metrics = eval_metrics

        # Log the overall win rate and average attempts
        self.log('test_win_rate', eval_metrics['win_rate'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('overall_avg_attempts',
                 eval_metrics['attempts'], on_step=False, on_epoch=True, prog_bar=True)

        # print("Overall Win Rate:", eval_metrics['win_rate'])
        # print("Overall Average Attempts:", eval_metrics['attempts'])

        # return eval_metrics

        # Update the data module's sampler with new performance metrics
        self.trainer.datamodule.update_train_dataloader(
            eval_metrics['length_wise_stats'])

    def evaluate_and_log(self):
        # Ensure the model is in evaluation mode
        self.eval()

        # Perform evaluation and gather statistics
        result = play_games_and_calculate_stats(self, self.test_words,
                                                self.char_frequency,
                                                self.max_word_length)

        # Extracting necessary statistics
        stats = result['stats']
        win_rate = result['overall_win_rate']
        attempts = result['overall_avg_attempts']
        length_wise_stats = result['length_wise_stats']

        # Switch back to training mode
        self.train()

        # Return the collected statistics for further use if needed
        return {'stats': stats, 'win_rate': win_rate,
                'attempts': attempts, 'length_wise_stats': length_wise_stats}

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     scheduler = {
    #         'scheduler': ReduceLROnPlateau(optimizer, mode='min',
    #                                        factor=0.1, patience=2, verbose=True),

    #         # 'val_miss_penalty'  # Name of the metric to monitor 'test_win_rate'
    #         # 'monitor': 'test_win_rate'
    #         'monitor': 'val_miss_penalty'
    #     }
    #     return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     return [self.optimizer], self.lr_schedulers

    def calculate_loss(self, outputs, labels, input_lens,
                       miss_chars, l1_factor=0.01, l2_factor=0.01):

        # # print(f"model out: ", model_out.shape)
        # # outputs = torch.sigmoid(model_out)
        # outputs = model_out

        # print(f"print output shape: ", outputs.shape)
        # print(f"label shape: ", labels.shape)
        # print(f"input_lens: ", input_lens)
        # print(f"miss_chars: ", miss_chars.shape)

        miss_penalty = torch.sum(outputs * miss_chars) / outputs.numel()
        # print(f"miss_penalty: ", miss_penalty)

        # Calculate weights for loss function
        weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)

        weights = weights_orig.unsqueeze(1).unsqueeze(
            2).expand(-1, outputs.size(1), -1)

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
