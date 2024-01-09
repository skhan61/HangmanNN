import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import SimpleLSTM
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from scr.feature_engineering import *
from scr.guess import guess as guess_fqn


class HangmanModel(pl.LightningModule):
    def __init__(self, lstm_model, learning_rate,
                 char_frequency, max_word_length):

        super().__init__()
        self.model = lstm_model
        self.learning_rate = learning_rate
        self.char_frequency = char_frequency
        self.max_word_length = max_word_length

        # self.save_hyperparameters()  # This line saves the hyperparameters

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

        loss, miss_penalty = self.calculate_loss(outputs,
                                                 encoded_guess, original_seq_lengths_tensor, batch_missed_chars)

        # If you know the batch size, explicitly provide it when logging
        # Or however you can determine the batch size
        batch_size = len(batch['guessed_states'])
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
            state, actual_guess, full_word = batch
            guessed_letters = []

            predicted_guess = guess_fqn(self.model, state[0],
                                        self.char_frequency, self.max_word_length, guessed_letters)

            fets, missed_chars = process_batch_of_games(
                [state], self.char_frequency, self.max_word_length, max_seq_length=1)
            fets, missed_chars = fets.to(self.device), missed_chars.to(self.device)
            seq_lens = torch.tensor([fets.size(1)], dtype=torch.long, device=self.device)

            outputs = self(fets, seq_lens, missed_chars)
            encoded_guess = pad_and_reshape_labels([actual_guess], max_seq_length=1).to(self.device)

            loss, miss_penalty = self.calculate_loss(outputs, encoded_guess, seq_lens, missed_chars)

            self.predicted_guesses.append(predicted_guess)
            self.actual_guesses.append(actual_guess)

                    # Calculate and log metrics every 32 states
            if len(self.predicted_guesses) >= 32:
                accuracy = accuracy_score(self.actual_guesses, self.predicted_guesses)
                f1 = f1_score(self.actual_guesses, self.predicted_guesses, average='weighted')

                # Directly specify batch size when logging
                self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=32)
                self.log('val_f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=32)

                self.predicted_guesses = []
                self.actual_guesses = []

            return {'loss': loss, 'miss_penalty': miss_penalty}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def calculate_loss(self, outputs, labels, input_lens, miss_chars):
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

        return loss, miss_penalty


# import pytorch_lightning as pl
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# from model import SimpleLSTM
# from torch.utils.data import DataLoader

# from scr.feature_engineering import *


# class HangmanModel(pl.LightningModule):
#     def __init__(self, lstm_model, learning_rate,
#                  char_frequency, max_word_length):

#         super().__init__()
#         self.model = lstm_model
#         self.learning_rate = learning_rate
#         self.char_frequency = char_frequency
#         self.max_word_length = max_word_length

#     def forward(self, fets, original_seq_lens, missed_chars):
#         return self.model(fets, original_seq_lens, missed_chars)

#     def training_step(self, batch, batch_idx):
#         states = batch['guessed_states']
#         guesses = batch['guessed_letters']
#         max_seq_length = batch['max_seq_len']
#         original_seq_lengths = batch['original_seq_lengths']

#         batch_features, batch_missed_chars = process_batch_of_games(
#             states, self.char_frequency, self.max_word_length, max_seq_length)

#         original_seq_lengths_tensor = torch.tensor(original_seq_lengths,
#                                                    dtype=torch.long, device=self.device)

#         encoded_guess = pad_and_reshape_labels(guesses, max_seq_length)
#         encoded_guess = encoded_guess.to(self.device)  # Move to correct device

#         # Move batch_missed_chars to the correct device
#         batch_features = batch_features.to(self.device)
#         batch_missed_chars = batch_missed_chars.to(self.device)

#         outputs = self(
#             batch_features, original_seq_lengths_tensor, batch_missed_chars)

#         loss, miss_penalty = self.calculate_loss(outputs,
#                 encoded_guess, original_seq_lengths_tensor, batch_missed_chars)

#         # If you know the batch size, explicitly provide it when logging
#         # Or however you can determine the batch size
#         batch_size = len(batch['guessed_states'])
#         self.log('train_loss', loss, on_step=True, on_epoch=True,
#                  prog_bar=True, batch_size=batch_size)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         state, guess, full_word = batch

#         fets, missed_chars = process_batch_of_games(
#             [state], self.char_frequency, self.max_word_length, max_seq_length=1)

#         fets, missed_chars = fets.to(self.device), missed_chars.to(
#             self.device)  # Move tensors to correct device

#         seq_lens = torch.tensor(
#             [fets.size(1)], dtype=torch.long, device=self.device)


#         outputs = self(fets, seq_lens, missed_chars)

#         print(outputs.shape)

#         encoded_guess = pad_and_reshape_labels([guess], max_seq_length=1)
#         encoded_guess = encoded_guess.to(self.device)  # Move to correct device

#         loss, miss_penalty = self.calculate_loss(
#             outputs, encoded_guess, seq_lens, missed_chars)

#         batch_size = len(state) if isinstance(state, list) else state.size(0)
#         self.log('val_loss', loss, on_step=True, on_epoch=True,
#                  prog_bar=True, batch_size=batch_size)

#         return {'loss': loss, 'miss_penalty': miss_penalty}

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer

#     def calculate_loss(self, model_out, labels, input_lens, miss_chars):
#         # print(f"model out: ", model_out.shape)
#         outputs = torch.sigmoid(model_out)

#         # print(f"print output shape: ", outputs.shape)
#         # print(f"label shape: ", labels.shape)
#         # print(f"input_lens: ", input_lens)
#         # print(f"miss_chars: ", miss_chars.shape)

#         miss_penalty = torch.sum(outputs * miss_chars) / outputs.numel()
#         # print(f"miss_penalty: ", miss_penalty)

#         # Calculate weights for loss function
#         weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)

#         weights = weights_orig.unsqueeze(1).unsqueeze(
#             2).expand(-1, model_out.size(1), -1)
#         # print(f"weights shape: ", weights.shape)

#         loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
#         loss = loss_func(model_out, labels)

#         # Ensure seq_lens_mask is on the same device as model_out
#         seq_lens_mask = torch.arange(model_out.size(1), device=model_out.device).expand(
#             len(input_lens), model_out.size(1)) < input_lens.unsqueeze(1)
#         # print(f"seq_lens_mask shape: ", seq_lens_mask.shape)

#         loss = loss * seq_lens_mask.unsqueeze(-1).float()
#         loss = loss.sum() / seq_lens_mask.sum()
#         # print(f"final loss: ", loss)

#         return loss, miss_penalty
