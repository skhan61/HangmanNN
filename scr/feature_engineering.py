import gc
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

# 1. Common Utility Functions
# Character set and mapping
char_to_idx = {char: idx for idx, char in
               enumerate(['_'] + list('abcdefghijklmnopqrstuvwxyz'))}

idx_to_char = {idx: char for char, idx in char_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_word(word):
    return [char_to_idx[char] for char in word]


def get_missed_characters(word):
    # Check if the input is a list and extract the first element if so
    if isinstance(word, list) and len(word) > 0:
        word = word[0]

    # print(word)
    all_chars = set(char_to_idx.keys())
    present_chars = set(word)
    missed_chars = all_chars - present_chars
    return torch.tensor([1 if char in missed_chars else
                         0 for char in char_to_idx],
                        dtype=torch.float)


def calculate_char_frequencies(word_list):
    char_counts = Counter(''.join(word_list))
    return {char: char_counts[char] / sum(char_counts.values())
            for char in char_to_idx}


def pad_tensor(tensor, length):
    return torch.cat([tensor, torch.zeros(length - len(tensor))], dim=0)


def encode_ngrams(ngrams, n):
    # print(f"Encoding n-grams: {ngrams}")
    encoded_ngrams = [char_to_idx[char] for ngram in ngrams for char in ngram]
    fixed_length = n * 2
    encoded = encoded_ngrams[:fixed_length] + \
        [0] * (fixed_length - len(encoded_ngrams))
    # print(f"Encoded n-grams (fixed length): {encoded}")
    return encoded


def calculate_word_frequencies(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}


def extract_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]


# 3. Inference Data Preparation
# Dont not change

def encode_guess(self, guess):
    # Assuming guess is a single character
    return char_to_idx.get(guess, char_to_idx['_'])


def process_single_word_inference(word, char_frequency,
                                  max_word_length):
    feature_set = build_feature_set(word, char_frequency, max_word_length)
    missed_chars = get_missed_characters(word)
    # Remove batch dimension and return
    return feature_set.squeeze(0), missed_chars


def process_game_sequence(game_states, char_frequency,
                          max_word_length, max_seq_length):
    # Calculate the number of features using the first game state
    sample_features = build_feature_set(
        game_states[0], char_frequency, max_word_length)
    # Assuming the last dimension holds the features
    num_features = sample_features.shape[-1]

    # Initialize the tensors for sequence features and missed characters
    sequence_features = torch.zeros(
        max_seq_length, max_word_length * num_features)

    sequence_missed_chars = torch.zeros(max_seq_length, len(char_to_idx))

    for i, state in enumerate(game_states):
        if i < max_seq_length:
            state_features, missed_chars = process_single_word_inference(
                state, char_frequency, max_word_length)
            sequence_features[i] = state_features.view(-1)
            sequence_missed_chars[i] = missed_chars

    return sequence_features, sequence_missed_chars


def process_batch_of_games(batch_of_games, char_frequency,
                           max_word_length, max_seq_length):
    # Define batch_size as the number of games in the batch
    batch_size = len(batch_of_games)

    # Check the number of features using the first state of the first game
    sample_features = build_feature_set(
        batch_of_games[0][0], char_frequency, max_word_length)

    # Assuming the last dimension holds the features
    num_features = sample_features.shape[-1]

    # Initialize tensors for batch features and missed characters
    batch_features = torch.zeros(
        batch_size, max_seq_length, max_word_length * num_features)
    batch_missed_chars = torch.zeros(
        batch_size, max_seq_length, len(char_to_idx))

    for i, game_states in enumerate(batch_of_games):
        sequence_features, sequence_missed_chars = process_game_sequence(
            game_states, char_frequency, max_word_length, max_seq_length)
        batch_features[i] = sequence_features
        batch_missed_chars[i] = sequence_missed_chars

    return batch_features, batch_missed_chars


def build_feature_set(word, char_frequency, max_word_length, ngram_n=3, normalize=True):
    word_len = len(word)

    # Encode the word
    encoded_word = encode_word(word)

    # Features
    word_length_feature = [word_len / max_word_length] * word_len
    positional_feature = [pos / max_word_length for pos in range(word_len)]
    frequency_feature = [char_frequency.get(idx_to_char.get(
        char_idx, '_'), 0) for char_idx in encoded_word]

    # N-grams feature
    ngrams = extract_ngrams(word, ngram_n)
    ngram_feature = encode_ngrams(ngrams, ngram_n)

    # Truncate or pad ngram feature to match word length
    ngram_feature = ngram_feature[:word_len] + \
        [0] * (word_len - len(ngram_feature))

    # Normalizing features if required
    if normalize:
        max_freq = max(char_frequency.values()) if char_frequency else 1
        frequency_feature = [freq / max_freq for freq in frequency_feature]

        max_ngram_idx = max(char_to_idx.values())
        ngram_feature = [(ngram_idx / max_ngram_idx)
                         for ngram_idx in ngram_feature]

    # Combine the features
    combined_features = [
        torch.tensor(encoded_word, dtype=torch.long),
        torch.tensor(word_length_feature, dtype=torch.float),
        torch.tensor(positional_feature, dtype=torch.float),
        torch.tensor(frequency_feature, dtype=torch.float),
        torch.tensor(ngram_feature, dtype=torch.float)
    ]

    # Stack and pad/truncate to max_word_length
    features_stacked = torch.stack(combined_features, dim=1)

    # Pad or truncate the features to match max_word_length
    if word_len < max_word_length:
        padding = max_word_length - word_len
        features_padded = F.pad(
            features_stacked, (0, 0, 0, padding), "constant", 0)
    else:
        features_padded = features_stacked[:max_word_length, :]

    return features_padded  # Only return the feature tensor


# Correct the extract_ngrams function


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


def process_single_game_state(game_state, char_frequency, max_word_length):
    # print("Game state received:", game_state)  # Debugging statement
    current_state, guessed_characters = game_state

    # Process this single game state
    sequence_features, sequence_missed_chars = process_game_sequence(
        [current_state], char_frequency, max_word_length, 1)  # max_seq_length is 1 for single game state

    # Since it's a single game state, we extract the first element from the batch
    return sequence_features[0], sequence_missed_chars[0]


gc.collect()
