import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import SimpleLSTM
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from scr.feature_engineering import *
from scr.guess import guess as guess_fqn

# # Example scheduler - adjust parameters as needed
# scheduler = ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.1, patience=2, verbose=True)


class HangmanModel(pl.LightningModule):
    def __init__(self, lstm_model, learning_rate, char_frequency,
                 max_word_length, l1_factor=0.01, l2_factor=0.01):
        super().__init__()
        self.model = lstm_model
        self.learning_rate = learning_rate
        self.char_frequency = char_frequency
        self.max_word_length = max_word_length

        # Regularization factors
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

        # Predicted and actual guesses for metrics calculation
        self.predicted_guesses = []
        self.actual_guesses = []

    def forward(self, fets, original_seq_lens, missed_chars):
        return self.model(fets, original_seq_lens, missed_chars)

    def training_step(self, batch, batch_idx):
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        batch_features, batch_missed_chars = process_batch_of_games(
            states, self.char_frequency, self.max_word_length, max_seq_length)

        original_seq_lengths_tensor = torch.tensor(original_seq_lengths,
                                                   dtype=torch.long, device=self.device)

        encoded_guess = pad_and_reshape_labels(guesses, max_seq_length)
        encoded_guess = encoded_guess.to(self.device)  # Move to correct device

        # Move batch_missed_chars to the correct device
        batch_features = batch_features.to(self.device)
        batch_missed_chars = batch_missed_chars.to(self.device)

        outputs = self(
            batch_features, original_seq_lengths_tensor, batch_missed_chars)

        # loss, miss_penalty = self.calculate_loss(outputs,
        #                                          encoded_guess, original_seq_lengths_tensor, batch_missed_chars)

        # Calculate loss with regularization
        loss, miss_penalty = self.calculate_loss(outputs, encoded_guess, original_seq_lengths_tensor,
                                                 batch_missed_chars, self.l1_factor, self.l2_factor)

        # If you know the batch size, explicitly provide it when logging
        # Or however you can determine the batch size
        batch_size = len(batch['guessed_states'])
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        # Process the batch
        batch_features, batch_missed_chars = process_batch_of_games(
            states, self.char_frequency, self.max_word_length, max_seq_length)

        original_seq_lengths_tensor = torch.tensor(original_seq_lengths,
                                                   dtype=torch.long, device=self.device)

        encoded_guess = pad_and_reshape_labels(guesses, max_seq_length)
        encoded_guess = encoded_guess.to(self.device)  # Move to correct device

        # Move batch_missed_chars to the correct device
        batch_features = batch_features.to(self.device)
        batch_missed_chars = batch_missed_chars.to(self.device)

        # Forward pass
        outputs = self(
            batch_features, original_seq_lengths_tensor, batch_missed_chars)

        # Calculate loss without the regularization factors
        loss, miss_penalty = self.calculate_loss(outputs, encoded_guess,
                                                 original_seq_lengths_tensor,
                                                 batch_missed_chars)

        # Determine the batch size for logging purposes
        # or any other way to determine the batch size
        batch_size = len(states)

        # Logging the validation loss and miss penalty with batch_size
        self.log('val_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)
        self.log('val_miss_penalty', miss_penalty, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=batch_size)

        return {'val_loss': loss, 'val_miss_penalty': miss_penalty}

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    #     # Define the learning rate scheduler
    #     scheduler = {
    #         'scheduler': lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
    #         'name': 'learning_rate',
    #         'interval': 'epoch',
    #         'frequency': 1
    #     }

    #     return [optimizer], [scheduler]

    #     # return optimizer

    # Add the scheduler to the LightningModule configuration
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True),
            'monitor': 'val_loss'  # Name of the metric to monitor
        }
        return [optimizer], [scheduler]

    def calculate_loss(self, outputs, labels, input_lens,
                       miss_chars, l1_factor=0.0, l2_factor=0.0):
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
