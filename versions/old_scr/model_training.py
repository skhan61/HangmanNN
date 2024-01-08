import random
from collections import defaultdict
from multiprocessing import Pool

import torch
import torch.nn.functional as F
import torch.optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from scr.feature_engineering import *
from scr.game import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_on_data_loader(model, data_loader,
                         device, l1_lambda=0.001,
                         l2_lambda=0.001):
    model.train()
    model.to(device)
    total_loss = 0
    total_miss_penalty = 0
    total_batches = 0

    optimizer = model.optimizer

    for batch in data_loader:
        states = batch['guessed_states']
        guesses = batch['guessed_letters']
        max_seq_length = batch['max_seq_len']
        original_seq_lengths = batch['original_seq_lengths']

        # Preprocessing: Process the batch data
        batch_features, batch_missed_chars = process_batch_of_games(
            states, guesses, char_frequency, max_word_length, max_seq_length)

        # Move the tensors to the specified device
        batch_features = batch_features.to(device)
        batch_missed_chars = batch_missed_chars.to(device)

        outputs = model(game_states_batch, lengths_batch, missed_chars_batch)
        
        reshaped_labels = pad_and_reshape_labels(
            labels_batch, outputs.shape).to(device)

        loss, miss_penalty = model.calculate_loss(
            outputs, reshaped_labels, lengths_batch, missed_chars_batch, 27)

        # Normalization of loss components
        normalized_loss = loss  # / (loss + 1e-6)
        normalized_miss_penalty = 2 * miss_penalty

        # # Debug: Print the loss and miss penalty every N batches
        # if batch_idx % 50 == 0:  # Adjust N according to your preference
        #     print(
        #         f"Batch {batch_idx}: Loss - {loss.item()}, Miss Penalty - {miss_penalty.item()}")
        #     print(
        #         f"Normalized Loss - {normalized_loss.item()}, Normalized Miss Penalty - {normalized_miss_penalty.item()}")

        # Regularization terms
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        # Apply conditional regularization based on optimizer type
        if optimizer_type == 'Adam':
            total_loss_with_reg = normalized_loss + \
                normalized_miss_penalty + l1_lambda * l1_norm
        elif optimizer_type == 'SGD':
            total_loss_with_reg = normalized_loss + normalized_miss_penalty + \
                l1_lambda * l1_norm + l2_lambda * l2_norm
        else:
            total_loss_with_reg = normalized_loss + normalized_miss_penalty

        optimizer.zero_grad()
        total_loss_with_reg.backward()
        optimizer.step()

        total_loss += loss.item()
        total_miss_penalty += miss_penalty.item()
        total_batches += 1

    average_loss = total_loss / total_batches if total_batches > 0 else 0
    average_miss_penalty = total_miss_penalty / \
        total_batches if total_batches > 0 else 0

    return average_loss, average_miss_penalty


def validate_hangman(model, data_loader, char_frequency, max_word_lengths,
                     device, unique_words_set, max_attempts=6, normalize=True,
                     max_games_per_epoch=1000):

    char_loss_total, char_miss_penalty_total = 0, 0
    game_wins_total, game_attempts_total, game_total_count = 0, 0, 0
    game_word_statistics = {}
    game_length_statistics = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_attempts": 0, "games": 0})

    # Shuffle the unique words to randomize selection
    unique_words = list(unique_words_set)
    random.shuffle(unique_words)

    total_games_played = 0

    # New variable for detailed miss penalty tracking
    total_incorrect_guesses = 0
    miss_penalty_per_game = [] 

    with torch.no_grad():
        for batch in data_loader:

            state, guess = batch[0][0], batch[0][1]
            word = batch[1]
            # print(state, guess)

            batch_features, batch_missed_chars = process_batch_of_games([state],
                        [guess],
                        char_frequency,
                        max_word_length,
                        1)

            batch_features, batch_missed_chars, batch_labels \
                = batch_features.to(
                device), batch_missed_chars.to(device), batch_labels.to(device)

            sequence_lengths = torch.tensor([batch_features.size(
                1)] * batch_features.size(0), dtype=torch.long).cpu()

            outputs = model(batch_features, sequence_lengths,
                            batch_missed_chars)

            reshaped_labels = pad_and_reshape_labels(
                batch_labels, outputs.shape).to(device)

            loss, miss_penalty = model.calculate_loss(
                outputs, reshaped_labels, sequence_lengths, batch_missed_chars, 27)

            char_loss_total += loss.item()
            char_miss_penalty_total += miss_penalty.item()

            # Play games only if the limit has not been reached
            while total_games_played < max_games_per_epoch and unique_words:
                full_word = unique_words.pop()
                word_length = len(full_word)

                won, final_word, attempts = play_game_with_a_word(
                    model, full_word, char_frequency, max_word_lengths, device, \
                        max_attempts, normalize)

                game_word_statistics[full_word] = {
                    "won": won, "final_word": final_word, "attempts": attempts}
                game_length_statistics[word_length]["games"] += 1
                game_length_statistics[word_length]["total_attempts"] += attempts

                if won:
                    game_length_statistics[word_length]["wins" ] += 1
                    game_wins_total += 1
                else:
                    game_length_statistics[word_length]["losses"] += 1

                game_attempts_total += attempts
                game_total_count += 1
                total_games_played += 1

    avg_char_loss = char_loss_total / \
        len(data_loader) if len(data_loader) > 0 else 0
    avg_char_miss_penalty = char_miss_penalty_total / \
        len(data_loader) if len(data_loader) > 0 else 0

    game_win_rate = game_wins_total / game_total_count \
        if game_total_count > 0 else 0

    avg_game_attempts = game_attempts_total / \
        game_total_count if game_total_count > 0 else 0

    return {
        "avg_loss": avg_char_loss,
        "avg_miss_penalty": avg_char_miss_penalty,

        "game_simulation": {
            "win_rate": game_win_rate,
            "average_attempts": avg_game_attempts,
            "total_games": game_total_count,
            "total_wins": game_wins_total,
            "total_losses": game_total_count - game_wins_total,
            "game_stats": game_word_statistics,
            "length_stats": dict(game_length_statistics)
        }
    }


# def validate_hangman(model, data_loader, char_frequency, max_word_lengths,
#                      device, unique_words_set, max_attempts=6, normalize=True,
#                      max_games_per_epoch=1000):
#     char_loss_total, char_miss_penalty_total = 0, 0
#     game_wins_total, game_attempts_total, game_total_count = 0, 0, 0
#     game_word_statistics = {}
#     game_length_statistics = defaultdict(
#         lambda: {"wins": 0, "losses": 0, "total_attempts": 0, "games": 0})

#     # New variable for detailed miss penalty tracking
#     total_incorrect_guesses = 0

#     # Shuffle the unique words to randomize selection
#     unique_words = list(unique_words_set)
#     random.shuffle(unique_words)

#     total_games_played = 0

#     with torch.no_grad():
#         for batch in data_loader:
#             batch_features, batch_missed_chars, batch_labels, _ = batch
#             batch_features, batch_missed_chars, batch_labels = batch_features.to(
#                 device), batch_missed_chars.to(device), batch_labels.to(device)

#             sequence_lengths = torch.tensor([batch_features.size(
#                 1)] * batch_features.size(0), dtype=torch.long).cpu()
#             outputs = model(batch_features, sequence_lengths,
#                             batch_missed_chars)
#             reshaped_labels = pad_and_reshape_labels(
#                 batch_labels, outputs.shape).to(device)

#             loss, miss_penalty = model.calculate_loss(
#                 outputs, reshaped_labels, sequence_lengths, batch_missed_chars, 27)

#             char_loss_total += loss.item()
#             char_miss_penalty_total += miss_penalty.item()

#             # Play games only if the limit has not been reached
#             while total_games_played < max_games_per_epoch and unique_words:
#                 full_word = unique_words.pop()
#                 word_length = len(full_word)

#                 won, final_word, attempts = play_game_with_a_word(
#                     model, full_word, char_frequency, max_word_lengths, device, max_attempts, normalize)

#                 # Track incorrect guesses
#                 incorrect_guesses = sum(1 for char in set(
#                     final_word) if char not in full_word)
#                 total_incorrect_guesses += incorrect_guesses

#                 game_word_statistics[full_word] = {
#                     "won": won, "final_word": final_word, "attempts": attempts, "incorrect_guesses": incorrect_guesses}
#                 game_length_statistics[word_length]["games"] += 1
#                 game_length_statistics[word_length]["total_attempts"] += attempts

#                 if won:
#                     game_length_statistics[word_length]["wins"] += 1
#                     game_wins_total += 1
#                 else:
#                     game_length_statistics[word_length]["losses"] += 1

#                 game_attempts_total += attempts
#                 game_total_count += 1
#                 total_games_played += 1

#     avg_char_loss = char_loss_total / \
#         len(data_loader) if len(data_loader) > 0 else 0
#     avg_char_miss_penalty = char_miss_penalty_total / \
#         len(data_loader) if len(data_loader) > 0 else 0
#     game_win_rate = game_wins_total / game_total_count if game_total_count > 0 else 0
#     avg_game_attempts = game_attempts_total / \
#         game_total_count if game_total_count > 0 else 0
#     avg_game_miss_penalty = total_incorrect_guesses / \
#         game_total_count if game_total_count > 0 else 0

#     return {
#         "avg_loss": avg_char_loss,
#         "avg_miss_penalty": avg_char_miss_penalty,
#         "game_simulation": {
#             "win_rate": game_win_rate,
#             "average_attempts": avg_game_attempts,
#             "average_miss_penalty": avg_game_miss_penalty,
#             "total_games": game_total_count,
#             "total_wins": game_wins_total,
#             "total_losses": game_total_count - game_wins_total,
#             "game_stats": game_word_statistics,
#             "length_stats": dict(game_length_statistics)
#         }
#     }
