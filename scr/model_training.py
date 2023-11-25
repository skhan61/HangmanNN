from collections import defaultdict
from scr.game import *
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from scr.feature_engineering import *

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
            print("Encountered an empty batch")
            continue  # Skip empty batches

        game_states_batch, lengths_batch, missed_chars_batch, labels_batch = batch
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


# Example of what one validation sample would look like.
# Here, the sample represents the game state "_pp_e" (after guessing 'p'),
# the next guess 'l', and the full word "apple".

# # Validation Sample: (['_pp_e', 'p'], 'l', 'apple')
def validate_hangman(model, data_loader, char_frequency,
                     max_word_length, device, max_attempts=6,
                     normalize=True):
    # Initialize metrics for character-level validation
    total_loss, total_miss_penalty, correct_char_predictions = 0, 0, 0
    correct_word_predictions, total_char_predictions, total_words = 0, 0, 0
    word_statistics = {}

    # Initialize metrics for game simulation
    total_wins, total_attempts, total_games = 0, 0, 0
    word_stats, word_length_stats = {}, defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_attempts": 0, "games": 0})

    # with torch.no_grad():
    #     for batch in data_loader:
    with torch.no_grad():
        for batch in data_loader:
            batch_features_tensor, \
                batch_missed_chars_tensor, batch_labels_tensor, \
                batch_full_words = batch
            # Move all tensors to the specified device
            batch_features_tensor = batch_features_tensor.to(device)
            batch_missed_chars_tensor = batch_missed_chars_tensor.to(device)
            batch_labels_tensor = batch_labels_tensor.to(device)

            # Derive batch size and max sequence length from the data
            batch_size = batch_features_tensor.size(0)
            max_seq_length = batch_features_tensor.size(1)

            sequence_lengths = torch.tensor(
                [max_seq_length] * batch_size, dtype=torch.long).cpu()

            # Model's inference
            outputs = model(batch_features_tensor,
                            sequence_lengths, batch_missed_chars_tensor)

            # Reshape labels to match model output
            reshaped_labels = pad_and_reshape_labels(
                batch_labels_tensor, outputs.shape).to(device)

            # Calculate loss and miss penalty
            loss, miss_penalty = model.calculate_loss(
                outputs, reshaped_labels, sequence_lengths,
                batch_missed_chars_tensor, 27)
            total_loss += loss.item()
            total_miss_penalty += miss_penalty.item()

            # Compute character-level accuracy
            # Assuming your model outputs class probabilities for each character
            predicted_chars = outputs.argmax(dim=-1)  # .to(device)

            correct_char_predictions += (predicted_chars ==
                                         batch_labels_tensor).sum().item()

            total_char_predictions += batch_labels_tensor.nelement()

            # print(batch_full_words)

            # Compute word-level accuracy and detailed statistics
            for idx, full_word in enumerate(batch_full_words):
                next_char_label = batch_labels_tensor[idx][0]
                if next_char_label.ndim > 0:
                    next_char_label = next_char_label.item()

                next_char_prediction = predicted_chars[idx, -1].item()

                correct_prediction = int(
                    next_char_prediction == next_char_label)
                correct_word_predictions += correct_prediction
                total_words += 1

                # Update detailed statistics for the word
                if full_word not in word_statistics:
                    word_statistics[full_word] = {
                        "total_attempts": 0, "correct_attempts": 0,
                        "correctly_guessed": False}

                word_stats = word_statistics[full_word]
                word_stats["total_attempts"] += 1
                word_stats["correct_attempts"] += correct_prediction
                word_stats["correctly_guessed"] \
                    = word_stats["correct_attempts"] == len(full_word)

            # Game simulation for each word in the batch
            for full_word in batch_full_words:
                print(full_word)
                won, final_word, attempts = play_game_with_a_word(model, full_word,
                            char_frequency, max_word_length, device, max_attempts, normalize)

                # Update word-specific stats
                word_stats[full_word] = {
                    "won": won,
                    "final_word": final_word,
                    "attempts_used": attempts
                }

                # Update word length-specific stats
                word_length = len(full_word)
                word_length_stats[word_length]["games"] += 1
                word_length_stats[word_length]["total_attempts"] += attempts
                if won:
                    word_length_stats[word_length]["wins"] += 1
                else:
                    word_length_stats[word_length]["losses"] += 1

                # Update overall stats
                total_wins += int(won)
                total_attempts += attempts
                total_games += 1

    # Calculate win rate and average attempts
    win_rate = total_wins / total_games if total_games > 0 else 0
    avg_attempts = total_attempts / total_games if total_games > 0 else 0

    # Calculating detailed statistics for each word length
    for length, stats in word_length_stats.items():
        stats["win_rate"] = stats["wins"] / stats["games"]
        stats["avg_attempts"] = stats["total_attempts"] / stats["games"]

    # Combine metrics from both character-level validation and game simulation
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_miss_penalty = total_miss_penalty / \
        len(data_loader) if len(data_loader) > 0 else 0
    char_accuracy = correct_char_predictions / \
        total_char_predictions if total_char_predictions > 0 else 0
    word_accuracy = correct_word_predictions / total_words if total_words > 0 else 0

    return {
        "character_level": {
            "avg_loss": avg_loss,
            "avg_miss_penalty": avg_miss_penalty,
            "char_accuracy": char_accuracy,
            "word_accuracy": word_accuracy,
            "word_statistics": word_statistics
        },
        "game_simulation": {
            "win_rate": win_rate,
            "average_attempts": avg_attempts,
            "total_games": total_games,
            "total_wins": total_wins,
            "total_losses": total_games - total_wins,
            "word_stats": word_stats,
            "word_length_stats": dict(word_length_stats)
        }
    }

# def validate_hangman(model, data_loader, device):
#     model.eval()
#     total_loss = 0
#     total_miss_penalty = 0
#     correct_char_predictions = 0
#     correct_word_predictions = 0
#     total_char_predictions = 0
#     total_words = 0
#     word_statistics = {}  # Dictionary to store detailed statistics for each word

#     # with torch.no_grad():
#     #     for batch in data_loader:
#     with torch.no_grad():
#         for batch in data_loader:
#             batch_features_tensor, \
#                 batch_missed_chars_tensor, batch_labels_tensor, \
#                 batch_full_words = batch
#             # Move all tensors to the specified device
#             batch_features_tensor = batch_features_tensor.to(device)
#             batch_missed_chars_tensor = batch_missed_chars_tensor.to(device)
#             batch_labels_tensor = batch_labels_tensor.to(device)

#             # Derive batch size and max sequence length from the data
#             batch_size = batch_features_tensor.size(0)
#             max_seq_length = batch_features_tensor.size(1)

#             sequence_lengths = torch.tensor(
#                 [max_seq_length] * batch_size, dtype=torch.long).cpu()

#             # Model's inference
#             outputs = model(batch_features_tensor,
#                             sequence_lengths, batch_missed_chars_tensor)

#             # Reshape labels to match model output
#             reshaped_labels = pad_and_reshape_labels(
#                 batch_labels_tensor, outputs.shape).to(device)

#             # Calculate loss and miss penalty
#             loss, miss_penalty = model.calculate_loss(
#                 outputs, reshaped_labels, sequence_lengths,
#                 batch_missed_chars_tensor, 27)
#             total_loss += loss.item()
#             total_miss_penalty += miss_penalty.item()

#             # Compute character-level accuracy
#             # Assuming your model outputs class probabilities for each character
#             predicted_chars = outputs.argmax(dim=-1)  # .to(device)

#             correct_char_predictions += (predicted_chars ==
#                                          batch_labels_tensor).sum().item()

#             total_char_predictions += batch_labels_tensor.nelement()

#             # print(batch_full_words)

#             # Compute word-level accuracy and detailed statistics
#             for idx, full_word in enumerate(batch_full_words):
#                 next_char_label = batch_labels_tensor[idx][0]
#                 if next_char_label.ndim > 0:
#                     next_char_label = next_char_label.item()

#                 next_char_prediction = predicted_chars[idx, -1].item()

#                 correct_prediction = int(
#                     next_char_prediction == next_char_label)
#                 correct_word_predictions += correct_prediction
#                 total_words += 1

#                 # Update detailed statistics for the word
#                 if full_word not in word_statistics:
#                     word_statistics[full_word] = {
#                         "total_attempts": 0, "correct_attempts": 0,
#                         "correctly_guessed": False}

#                 word_stats = word_statistics[full_word]
#                 word_stats["total_attempts"] += 1
#                 word_stats["correct_attempts"] += correct_prediction
#                 word_stats["correctly_guessed"] \
#                     = word_stats["correct_attempts"] == len(full_word)

#     avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
#     avg_miss_penalty = total_miss_penalty / \
#         len(data_loader) if len(data_loader) > 0 else 0
#     char_accuracy = correct_char_predictions / \
#         total_char_predictions if total_char_predictions > 0 else 0
#     word_accuracy = correct_word_predictions / total_words if total_words > 0 else 0

#     return avg_loss, avg_miss_penalty, char_accuracy, \
#         word_accuracy, word_statistics


def pad_and_reshape_labels(labels, model_output_shape):
    batch_size, sequence_length, num_classes = model_output_shape

    # Calculate the total number of elements needed
    total_elements = batch_size * sequence_length

    # Pad the labels to the correct total length
    padded_labels = F.pad(input=labels, pad=(
        0, total_elements - labels.numel()), value=0)

    # Reshape the labels to match the batch and sequence length
    reshaped_labels = padded_labels.view(batch_size, sequence_length)

    # Convert to one-hot encoding
    one_hot_labels = F.one_hot(
        reshaped_labels, num_classes=num_classes).float()

    return one_hot_labels
