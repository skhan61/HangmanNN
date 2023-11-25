import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

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


def validate_hangman(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_miss_penalty = 0
    correct_char_predictions = 0
    correct_word_predictions = 0
    total_char_predictions = 0
    total_words = 0

    with torch.no_grad():
        for batch in data_loader:
            batch_features_tensor, batch_missed_chars_tensor, batch_labels_tensor, \
                batch_full_words = batch
            batch_features_tensor, batch_missed_chars_tensor = \
                batch_features_tensor.to(
                    device), batch_missed_chars_tensor.to(device)

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
                outputs, reshaped_labels, sequence_lengths, batch_missed_chars_tensor, 27)
            total_loss += loss.item()
            total_miss_penalty += miss_penalty.item()

            # Compute character-level accuracy
            # Assuming your model outputs class probabilities for each character
            predicted_chars = outputs.argmax(dim=-1)
            predicted_chars = predicted_chars.to(device)
            correct_char_predictions += (predicted_chars ==
                                         batch_labels_tensor).sum().item()
            total_char_predictions += batch_labels_tensor.nelement()

            # Compute word-level accuracy
            for idx, full_word in enumerate(batch_full_words):
                # Convert the sequence of predicted character indices to a word
                predicted_word = ''.join(
                    [idx_to_char[char_idx] for char_idx in predicted_chars[idx]])
                correct_word_predictions += int(predicted_word == full_word)
                total_words += 1

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_miss_penalty = total_miss_penalty / \
        len(data_loader) if len(data_loader) > 0 else 0
    char_accuracy = correct_char_predictions / \
        total_char_predictions if total_char_predictions > 0 else 0
    word_accuracy = correct_word_predictions / total_words if total_words > 0 else 0

    return avg_loss, avg_miss_penalty, char_accuracy, word_accuracy


# def validate_hangman(model, data_loader, device):
#     model.eval()
#     total_loss = 0
#     total_miss_penalty = 0
#     correct_char_predictions = 0
#     correct_word_predictions = 0
#     total_char_predictions = 0
#     total_words = 0

#     with torch.no_grad():
#         for batch in data_loader:
#             batch_features_tensor, batch_missed_chars_tensor, \
#                 batch_labels_tensor, batch_full_words = batch
#             batch_features_tensor, batch_missed_chars_tensor = \
#                 batch_features_tensor.to(
#                     device), batch_missed_chars_tensor.to(device)

#             # Derive batch size and max sequence length from the data
#             batch_size = batch_features_tensor.size(0)
#             max_seq_length = batch_features_tensor.size(1)

#             sequence_lengths = torch.tensor([max_seq_length]
#                                             * batch_size, dtype=torch.long).cpu()

#             # Model's inference
#             outputs = model(batch_features_tensor,
#                            sequence_lengths, batch_missed_chars_tensor)

#             # Reshape labels to match model output
#             reshaped_labels = pad_and_reshape_labels(batch_labels_tensor, outputs.shape)
#             print(reshaped_labels.shape)
#             reshaped_labels = reshaped_labels.to(device)

#                         # Calculate loss and miss penalty
#             loss, miss_penalty = model.calculate_loss(outputs, reshaped_labels, \
#                 sequence_lengths, batch_missed_chars_tensor, 27)
#             total_loss += loss.item()
#             total_miss_penalty += miss_penalty.item()

#             print(loss)
#             print(miss_penalty)


#             break

#     print(outputs.shape)


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
