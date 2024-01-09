import gc
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

# 1. Common Utility Functions
# Character set and mapping
# Add blank character '' to the character set
char_to_idx = {char: idx for idx, char in enumerate(
    ['', '_'] + list('abcdefghijklmnopqrstuvwxyz'))}

idx_to_char = {idx: char for char, idx in char_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_word(word):
    # print(word)
    return [char_to_idx[char] for char in word]


def get_missed_characters(word):
    # print(f'print word form get missed char: ', word)
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


def build_feature_set(word, char_frequency,
                      max_word_length, ngram_n=3, normalize=True):

    word_len = len(word)

    # Encode the word
    encoded_word = encode_word(word)

    # print(f'encoded words: ', encode_word)

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


def process_single_state(word, char_frequency, max_word_length):
    # Process a single game state (word) to get features and missed characters
    feature_set = build_feature_set(
        word, char_frequency, max_word_length)  # Get features for the word
    missed_chars = get_missed_characters(word)  # Get missed characters tensor
    # Return tensors for the single state
    return feature_set.squeeze(0), missed_chars


def process_game_sequence(game_states, char_frequency,
                          max_word_length, max_seq_length):
    # Process a sequence of game states
    num_features = build_feature_set(
        game_states[0], char_frequency, max_word_length).shape[-1]  # Determine number of features
    # Tensor for features of all states
    sequence_features = torch.zeros(
        max_seq_length, max_word_length * num_features)
    # Tensor for missed characters for all states
    sequence_missed_chars = torch.zeros(max_seq_length, len(char_to_idx))

    for i, state in enumerate(game_states):
        if i < max_seq_length:  # Process only up to the maximum sequence length
            state_features, missed_chars = process_single_state(
                state, char_frequency, max_word_length)
            # Store features in the tensor
            sequence_features[i] = state_features.view(-1)
            # Store missed characters in the tensor
            sequence_missed_chars[i] = missed_chars

    # Return tensors for the entire sequence
    return sequence_features, sequence_missed_chars


# def process_batch_of_games(guessed_states_batch,
#                            guessed_letters_batch, char_frequency,
#                            max_word_length, max_seq_length):

def process_batch_of_games(guessed_states_batch, char_frequency,
                           max_word_length, max_seq_length):


    # # def process_batch_of_games(guessed_states_batch, char_frequency, max_word_length, max_seq_length):
    # batch_size = len(guessed_states_batch)  # Number of games in the batch
    # print(f"guessed_states_batch: {guessed_states_batch}")  # Debugging
    # print(f"First state in batch: {guessed_states_batch[0]}")  # Debugging

    # first_state_features = build_feature_set(guessed_states_batch[0], char_frequency, max_word_length)
    # print(f"Features of first state: {first_state_features}")  # Debugging
    # num_features = first_state_features.shape[-1]  # Determine number of features

    # # Rest of the function code...


    batch_size = len(guessed_states_batch)  # Number of games in the batch
    num_features = build_feature_set(
        guessed_states_batch[0][0], char_frequency, max_word_length).shape[-1]  # Determine number of features

    # Tensor for features of all games
    batch_features = torch.zeros(
        batch_size, max_seq_length, max_word_length * num_features)
    batch_missed_chars = torch.zeros(batch_size, max_seq_length, len(
        char_to_idx))  # Tensor for missed characters for all games

    for i in range(batch_size):
        game_states = guessed_states_batch[i]  # States for a single game
        # Letters guessed in a single game
        # game_letters = guessed_letters_batch[i]
        sequence_features, sequence_missed_chars = process_game_sequence(
            game_states, char_frequency, max_word_length, max_seq_length)
        # Store features for each game in the batch
        batch_features[i] = sequence_features
        # Store missed characters for each game in the batch
        batch_missed_chars[i] = sequence_missed_chars

    return batch_features, batch_missed_chars  # Return tensors for the entire batch


def pad_and_reshape_labels(guesses, max_seq_length,
                           num_classes=len(char_to_idx)):
    """
    Pad, encode, and reshape labels for one-hot encoding.

    :param guesses: List of strings (guesses).
    :param max_seq_length: Maximum sequence length.
    :param char_to_idx: Dictionary mapping characters to indices.
    :param num_classes: Number of classes (size of the character set).
    :return: One-hot encoded labels of shape [batch_size, sequence_length, num_classes].
    """
    batch_size = len(guesses)

    # Initialize a zero tensor for padded labels
    padded_labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

    # Pad and encode each label in the batch
    for i, guess in enumerate(guesses):
        # Convert guess to indices using char_to_idx
        guess_indices = [char_to_idx.get(
            char, char_to_idx['']) for char in guess]

        # Pad the encoded guess
        length = min(len(guess_indices), max_seq_length)
        padded_labels[i, :length] = torch.tensor(guess_indices[:length])

    # Convert to one-hot encoding
    one_hot_labels = F.one_hot(padded_labels, num_classes=num_classes).float()

    return one_hot_labels


# def process_single_game_state(game_state,
#                               char_frequency,
#                               max_word_length):
#     print("Game state received:", game_state)  # Debugging statement
#     current_state, guessed_characters = game_state[0], game_state[1]

#     # Process this single game state
#     sequence_features, sequence_missed_chars = process_game_sequence(
#         [current_state], char_frequency, max_word_length, 1)  # max_seq_length is 1 for single game state

#     # Since it's a single game state, we extract the first element from the batch
#     return sequence_features[0], sequence_missed_chars[0]


# # Dummy batch of game states similar to 'e__e__e'
# guessed_states_batch = [
#     ['e__e__', 'e_e_e_', 'ee_e__', 'eee_e_', 'eeeee_'],
#     ['_e__e_', '__e_e_', '_ee__e', '_eee_e', '_eeee_'],
#     ['e___e_', 'e__ee_', 'e_e_e_', 'ee__e_', 'eee_e_']
# ]

# # Dummy batch of guessed letters for each state
# guessed_letters_batch = [
#     ['a', 'b', 'c', 'd', 'e'],
#     ['f', 'g', 'h', 'i', 'j'],
#     ['k', 'l', 'm', 'n', 'o']
# ]


# # Dummy batch of game states similar to 'e__e__e'
# guessed_states_batch = [
#     ['e__e__'],
# ]

# # Dummy batch of guessed letters for each state
# guessed_letters_batch = [
#     ['a'],
# ]


# state_fets, state_miss_char = process_game_sequence(game_states, char_frequency,
#                           max_word_length, max_seq_length)

# print(f'state fets shape: ', state_fets.shape)
# print(f'state_miss_chars shape: ', state_miss_char.shape)

# print()
# # Process the dummy batch
# batch_features, batch_missed_chars = process_batch_of_games(
#     guessed_states_batch, guessed_letters_batch, char_frequency, max_word_length, max_seq_length)

# # Outputs
# print("Batch Features Shape:", batch_features.shape)  # Expected: [3, 5, num_features]
# print("Batch Missed Chars Shape:", batch_missed_chars.shape)  # Expected: [3, 5, len(char_to_idx)]

gc.collect()
