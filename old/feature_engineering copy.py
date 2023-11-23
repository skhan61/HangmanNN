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
    all_chars = set(char_to_idx.keys())
    present_chars = set(word)
    missed_chars = all_chars - present_chars
    return torch.tensor([1 if char in missed_chars else
                         0 for char in char_to_idx],
                        dtype=torch.float)


def calculate_char_frequencies(word_list):
    char_counts = Counter(''.join(word_list))
    return {char: char_counts[char] / sum(char_counts.values()) for char in char_to_idx}


def extract_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]

    encoded_ngrams = [char_to_idx[char] for ngram in ngrams for char in ngram]
    fixed_length = n * 2
    return encoded_ngrams[:fixed_length] + [0] * (fixed_length - len(encoded_ngrams))


def pad_tensor(tensor, length):
    return torch.cat([tensor, torch.zeros(length - len(tensor))], dim=0)


def encode_ngrams(ngrams, n):
    encoded_ngrams = [char_to_idx[char] for ngram in ngrams for char in ngram]
    fixed_length = n * 2
    return encoded_ngrams[:fixed_length] + [0] * (fixed_length - len(encoded_ngrams))


def calculate_word_frequencies(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}


# 3. Train/Inference Data Preparation

# def add_features_for_training(word, initial_state, char_frequency,
#                               max_word_length):

#     encoded_word = encode_word(word)
#     # encoded_initial_state = encode(initial_state)

#     # Build feature set
#     feature_set = build_feature_set(encoded_initial_state, char_frequency,
#                                     max_word_length)

#     missed_chars = get_missed_characters(word)

#     return feature_set, torch.tensor(label, dtype=torch.float), missed_chars

# # 3. Inference Data Preparation
# Dont not change

# 3. Inference Data Preparation
# Dont not change

def encode_guess(self, guess):
    # Assuming guess is a single character
    return char_to_idx.get(guess, char_to_idx['_'])


# def process_single_word_inference(word, char_frequency,
#                                   max_word_length):

#     # encoded_word = encode_word(word)
#     feature_set = build_feature_set(word, char_frequency,
#                                     max_word_length)  # , ngram_n, normalize=True)
#     missed_chars = get_missed_characters(word)
#     return feature_set, missed_chars.unsqueeze(0)

def process_single_word_inference(word, char_frequency, max_word_length):
    feature_set = build_feature_set(
        word, char_frequency, max_word_length)  # No batch dimension added
    missed_chars = get_missed_characters(word)
    # Remove batch dimension and return
    return feature_set.squeeze(0), missed_chars


# def process_game_sequence(game_states, char_frequency, max_word_length, max_seq_length):
#     sequence_features = []
#     sequence_missed_chars = []

#     for state in game_states:
#         state_features, missed_chars = process_single_word_inference(
#             state, char_frequency, max_word_length)
#         sequence_features.append(state_features)
#         sequence_missed_chars.append(missed_chars)

#     # Pad the sequences if they are shorter than max_seq_length
#     if len(game_states) < max_seq_length:
#         padding_length = max_seq_length - len(game_states)
#         pad_features = torch.zeros(
#             padding_length, 1, max_word_length, sequence_features[0].size(-1))
#         pad_missed_chars = torch.zeros(
#             padding_length, 1, sequence_missed_chars[0].size(-1))

#         sequence_features = torch.cat(
#             (torch.cat(sequence_features, dim=0), pad_features), dim=0)
#         sequence_missed_chars = torch.cat(
#             (torch.cat(sequence_missed_chars, dim=0), pad_missed_chars), dim=0)

#     return sequence_features, sequence_missed_chars

def process_game_sequence(game_states, char_frequency, max_word_length, max_seq_length):
    sequence_features = []
    sequence_missed_chars = []

    for state in game_states:
        state_features, missed_chars = process_single_word_inference(
            state, char_frequency, max_word_length)
        sequence_features.append(state_features)
        sequence_missed_chars.append(missed_chars)

    # Stack the features and missed chars for all states
    # [num_states, max_word_length, features]
    sequence_features = torch.stack(sequence_features, dim=0)
    # [num_states, miss_char_features]
    sequence_missed_chars = torch.stack(sequence_missed_chars, dim=0)

    # Pad the sequences if they are shorter than max_seq_length
    if len(game_states) < max_seq_length:
        padding_length = max_seq_length - len(game_states)
        pad_features = torch.zeros(
            padding_length, max_word_length, sequence_features.size(-1))
        pad_missed_chars = torch.zeros(
            padding_length, sequence_missed_chars.size(-1))

        sequence_features = torch.cat((sequence_features, pad_features), dim=0)
        sequence_missed_chars = torch.cat(
            (sequence_missed_chars, pad_missed_chars), dim=0)

    # Add batch dimension
    return sequence_features.unsqueeze(0), sequence_missed_chars.unsqueeze(0)


def process_batch_of_games(batch_of_games,
                           char_frequency, max_word_length, max_seq_length):
    batch_features = []
    batch_missed_chars = []

    for game_states in batch_of_games:
        game_features, game_missed_chars = process_game_sequence(
            game_states, char_frequency, max_word_length, max_seq_length)
        batch_features.append(game_features)
        batch_missed_chars.append(game_missed_chars)

    return torch.stack(batch_features), torch.stack(batch_missed_chars)


def build_feature_set(word, char_frequency,
                      max_word_length, ngram_n=3,
                      normalize=True):
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

    return features_padded.unsqueeze(0)  # Adding batch dimension
    # features = [
    #     torch.tensor(encoded_word, dtype=torch.long),
    #     torch.tensor(word_length_feature, dtype=torch.float),
    #     torch.tensor(positional_feature, dtype=torch.float),
    #     torch.tensor(frequency_feature, dtype=torch.float),
    #     torch.tensor(ngram_feature, dtype=torch.float)
    # ]

    # return torch.stack(features, dim=1)


gc.collect()
