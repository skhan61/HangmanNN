from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from scr.feature_engineering import *
from scr.game import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_on_data_loader(model, data_loader, device, optimizer):
    model.train()  # Set the model to training mode
    model.to(device)
    total_loss = 0
    total_miss_penalty = 0
    total_batches = 0

    for batch in data_loader:
        if batch[0] is None:
            # print("Encountered an empty batch")
            continue  # Skip empty batches

        game_states_batch, lengths_batch, missed_chars_batch, labels_batch, _ = batch
        game_states_batch, lengths_batch, missed_chars_batch \
            = game_states_batch.to(device), \
            lengths_batch, missed_chars_batch.to(device)

        # Assuming 'model' is your trained model
        outputs = model(game_states_batch, lengths_batch, missed_chars_batch)

        # Reshape labels to match model output
        reshaped_labels = pad_and_reshape_labels(labels_batch, outputs.shape)
        reshaped_labels = reshaped_labels.to(device)

        # Compute loss and miss penalty
        loss, miss_penalty = model.calculate_loss(outputs, reshaped_labels,
                                                  lengths_batch, missed_chars_batch, 27)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_miss_penalty += miss_penalty.item()
        total_batches += 1

    average_loss = total_loss / total_batches if total_batches > 0 else 0
    average_miss_penalty = total_miss_penalty / \
        total_batches if total_batches > 0 else 0
    return average_loss, average_miss_penalty


def validate_hangman(model, data_loader, char_frequency, max_word_lengths,
                     device, max_attempts=6, normalize=True,
                     max_games_per_epoch=1000):
    char_loss_total, char_miss_penalty_total, char_correct_preds_total = 0, 0, 0
    char_correct_word_preds_total, char_total_preds, char_total_words = 0, 0, 0
    char_word_statistics = defaultdict(
        lambda: {"total_attempts": 0, "correct_attempts": 0, "correctly_guessed": False})

    game_wins_total, game_attempts_total, game_total_count = 0, 0, 0
    game_word_statistics = {}
    game_length_statistics = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_attempts": 0, "games": 0})

    # Collect unique words from the data loader
    unique_words = set()
    for _, _, _, batch_full_words in data_loader:
        unique_words.update(batch_full_words)

    # Shuffle the unique words to randomize selection
    unique_words = list(unique_words)
    random.shuffle(unique_words)

    total_games_played = 0

    with torch.no_grad():
        for batch in data_loader:
            batch_features, batch_missed_chars, batch_labels, _ = batch
            batch_features = batch_features.to(device)
            batch_missed_chars = batch_missed_chars.to(device)
            batch_labels = batch_labels.to(device)

            batch_size = batch_features.size(0)
            max_seq_length = batch_features.size(1)
            sequence_lengths = torch.tensor(
                [max_seq_length] * batch_size, dtype=torch.long).cpu()

            outputs = model(batch_features, sequence_lengths,
                            batch_missed_chars)
            reshaped_labels = pad_and_reshape_labels(
                batch_labels, outputs.shape).to(device)

            loss, miss_penalty = model.calculate_loss(
                outputs, reshaped_labels, sequence_lengths, batch_missed_chars, 27)
            char_loss_total += loss.item()
            char_miss_penalty_total += miss_penalty.item()

            predicted_chars = outputs.argmax(dim=-1)
            char_correct_preds_total += (predicted_chars ==
                                         batch_labels).sum().item()
            char_total_preds += batch_labels.nelement()

            for idx, full_word in enumerate(batch_full_words):
                next_char_label = batch_labels[idx][0].item()
                next_char_prediction = predicted_chars[idx, -1].item()
                correct_prediction = int(
                    next_char_prediction == next_char_label)
                char_correct_word_preds_total += correct_prediction
                char_total_words += 1

                char_word_statistics[full_word]["total_attempts"] += 1
                char_word_statistics[full_word]["correct_attempts"] += correct_prediction
                char_word_statistics[full_word]["correctly_guessed"] \
                    = char_word_statistics[full_word]["correct_attempts"] == len(
                    full_word)

            # Play games only if the limit has not been reached
            while total_games_played < max_games_per_epoch and unique_words:
                full_word = unique_words.pop()
                word_length = len(full_word)

                won, final_word, attempts = play_game_with_a_word(
                    model, full_word, char_frequency, max_word_length, device, max_attempts, normalize)
                game_word_statistics[full_word] = {
                    "won": won, "final_word": final_word, "attempts": attempts}
                game_length_statistics[word_length]["games"] += 1
                game_length_statistics[word_length]["total_attempts"] += attempts
                if won:
                    game_length_statistics[word_length]["wins"] += 1
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
    char_accuracy = char_correct_preds_total / \
        char_total_preds if char_total_preds > 0 else 0
    word_accuracy = char_correct_word_preds_total / \
        char_total_words if char_total_words > 0 else 0
    game_win_rate = game_wins_total / game_total_count if game_total_count > 0 else 0
    avg_game_attempts = game_attempts_total / \
        game_total_count if game_total_count > 0 else 0

    return {
        "character_level": {
            "avg_loss": avg_char_loss,
            "avg_miss_penalty": avg_char_miss_penalty,
            "char_accuracy": char_accuracy,
            "word_accuracy": word_accuracy,
            "word_stats": dict(char_word_statistics)
        },
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
