import gc
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *

gc.collect()


class ProcessedHangmanDataset(Dataset):
    def __init__(self, pkls_dir, char_freq_dict, max_word_length, mode='individual'):
        self.char_freq_dict = char_freq_dict
        self.max_word_length = max_word_length
        self.data = []  # Stores tuples of (game_state, label, additional_info)
        self.mode = mode

        for batch_dir in sorted(pkls_dir.iterdir(), key=lambda x: int(x.name) \
            if x.name.isdigit() else float('inf')):
            if batch_dir.is_dir():
                for pkl_file in batch_dir.glob("*.pkl"):
                    with open(pkl_file, 'rb') as file:
                        game_states = pickle.load(file)

                        parts = pkl_file.stem.split('_from_')
                        word, remaining = parts[0], parts[1].split('_')
                        initial_state, difficulty, outcome = '_'.join(
                            remaining[:-2]), remaining[-2], remaining[-1]

                        for game_state in game_states:
                            game_won, guesses = game_state
                            if len(guesses) > 0:
                                states, next_guesses = self.process_game_states(
                                    guesses)
                                additional_info = {'word': word, 'initial_state': initial_state,
                                                   'difficulty': difficulty, 'outcome': outcome}
                                self.data.append(
                                    (states, next_guesses, additional_info))

    def process_game_states(self, guesses):
        states = [guesses[0][1]]
        next_guesses = []
        for i in range(1, len(guesses)):
            next_guesses.append(guesses[i][0])
            states.append(guesses[i][1])

        if self.mode == 'individual':
            return states[:-1], next_guesses
        elif self.mode == 'entire_game':
            # Modify this if you need different logic
            return [states[0]], [states[-1]]
        else:
            raise ValueError("Invalid mode specified")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_states, labels, additional_info = self.data[idx]
        return game_states, labels, additional_info


# def custom_collate_fn(batch):
#     batch_features, batch_missed_chars, batch_labels = [], [], []
#     max_seq_length = 0  # Variable to track the maximum sequence length in the batch

#     # First, process each game to find the maximum sequence length
#     for item in batch:
#         game_states, labels, _ = item
#         if not game_states:
#             continue
#         max_seq_length = max(max_seq_length, len(game_states))

#     # Now, process each game again to pad sequences and collect batch data
#     for item in batch:
#         game_states, labels, _ = item
#         if not game_states:
#             continue

#         game_features, game_missed_chars = process_game_sequence(
#             game_states, char_frequency, max_word_length, len(game_states))

#         # Pad each game's features and missed characters to the maximum sequence length
#         if len(game_states) < max_seq_length:
#             padding_length = max_seq_length - len(game_states)

#             # Create padding tensor for game_features
#             padding_tensor_features = torch.zeros(
#                 padding_length, game_features.shape[1])
#             game_features_padded = torch.cat(
#                 [game_features, padding_tensor_features], dim=0)

#             # Create a separate padding tensor for game_missed_chars
#             padding_tensor_missed_chars = torch.zeros(
#                 padding_length, game_missed_chars.shape[1])
#             game_missed_chars_padded = torch.cat(
#                 [game_missed_chars, padding_tensor_missed_chars], dim=0)
#         else:
#             game_features_padded = game_features
#             game_missed_chars_padded = game_missed_chars

#         batch_features.append(game_features_padded)
#         batch_missed_chars.append(game_missed_chars_padded)
#         batch_labels.extend([char_to_idx[label] for label in labels])

#     # Before stacking, check if the lists are empty
#     if not batch_features or not batch_missed_chars:
#         # Handle empty batch here, maybe skip or return None
#         print("Encountered an empty batch")
#         return None, None, None

#     # Stack all games to form the batch
#     batch_features_stacked = torch.stack(batch_features)
#     batch_missed_chars_stacked = torch.stack(batch_missed_chars)
#     labels_tensor = torch.tensor(batch_labels, dtype=torch.long)

#     return batch_features_stacked, batch_missed_chars_stacked, labels_tensor
