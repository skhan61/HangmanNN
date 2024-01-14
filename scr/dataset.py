import os
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *

# class HangmanDataset(Dataset):
#     def __init__(self, parquet_files):
#         self.parquet_files = parquet_files


class HangmanDataset(Dataset):
    def __init__(self, parquet_files):
        # Ensure parquet_files is a list, even if it's a single path
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.lengths = [pd.read_parquet(f).shape[0]
                        for f in self.parquet_files]
        self.cumulative_lengths = self._compute_cumulative_lengths(
            self.lengths)

    def _compute_cumulative_lengths(self, lengths):
        cum_lengths = [0]
        total = 0
        for length in lengths:
            total += length
            cum_lengths.append(total)
        return cum_lengths

    def _get_file_and_local_idx(self, idx):
        for file_idx, cum_length in enumerate(self.cumulative_lengths):
            if idx < cum_length:
                local_idx = idx - self.cumulative_lengths[file_idx - 1]
                return self.parquet_files[file_idx - 1], local_idx
        raise IndexError("Index out of bounds")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        file_path, local_idx = self._get_file_and_local_idx(idx)
        df = pd.read_parquet(file_path)
        row = df.iloc[local_idx]

        # Process the row and return the necessary data
        return {
            'game_id': row['game_id'],
            'word': row['word'],
            'initial_state': row['initial_state'].split(','),
            'final_state': row['final_state'],
            'guessed_states': row['guessed_states'].split(','),
            'guessed_letters': row['guessed_letters'].split(','),
            'game_state': row['game_state'],
            'difficulty': row['difficulty'],
            'outcome': row['outcome'],
            'word_length': row['word_length'],
            'won': row['won'] == 'True'
        }


def custom_collate_fn(batch):
    # Since lengths are the same for states and letters
    max_seq_len = max(len(item['guessed_states']) for item in batch)
    padded_states = []
    padded_letters = []
    original_seq_lengths = []

    for item in batch:
        # Assuming both states and letters have the same length
        original_seq_len = len(item['guessed_states'])
        original_seq_lengths.append(original_seq_len)

        states_padding = [''] * (max_seq_len - original_seq_len)
        letters_padding = [''] * (max_seq_len - original_seq_len)

        padded_states.append(item['guessed_states'] + states_padding)
        padded_letters.append(item['guessed_letters'] + letters_padding)

    return {
        'guessed_states': padded_states,
        'guessed_letters': padded_letters,
        'max_seq_len': max_seq_len,
        'original_seq_lengths': original_seq_lengths
    }


class SimpleWordDataset(Dataset):
    """ Dataset for individual words """

    def __init__(self, word_list):
        self.word_list = word_list

    def __len__(self):
        return len(self.word_list)

    def __getitem__(self, idx):
        return self.word_list[idx]


# def create_validation_samples(game_data_list):
#     validation_samples = []
#     for game_data in game_data_list:
#         for i in range(len(game_data['guessed_states']) - 1):
#             current_state = game_data['guessed_states'][i]
#             next_guess = game_data['guessed_letters'][i + 1]
#             full_word = game_data['word']
#             validation_samples.append(([current_state, next_guess], full_word))

#     return validation_samples


# def validation_collate_fn(batch):
#     batch_states, batch_guesses, batch_full_words = [], [], []

#     for game_state, full_word in batch:
#         state = game_state[0]  # Current state
#         guess = game_state[1]  # Next guess

#         batch_states.append(state)
#         batch_guesses.append(guess)
#         batch_full_words.append(full_word)

#     # You can choose to convert lists to tensors if needed, or leave them as lists
#     return batch_states, batch_guesses, batch_full_words


# def create_val_loader(val_data):
#     val_samples = [create_validation_samples(
#         [game_data]) for game_data in val_data]
#     flattened_val_samples = [
#         sample for sublist in val_samples for sample in sublist]

#     val_loader = DataLoader(flattened_val_samples, batch_size=1,
#                             collate_fn=validation_collate_fn, shuffle=False)
#     return val_loader


# def validation_collate_fn(batch, char_frequency, max_word_length):
#     batch_features, batch_missed_chars, batch_labels, batch_full_words = [], [], [], []

#     for game_state, full_word in batch:
#         print(f'Game states from valid collate: ', game_state)
#         processed_state, missed_chars = process_batch_of_games(game_state,
#                                                                char_frequency,
#                                                                max_word_length,
#                                                                max_seq_length=1)

#         batch_features.append(processed_state.unsqueeze(0))
#         batch_missed_chars.append(missed_chars.unsqueeze(0))

#         encoded_word = encode_word(full_word)
#         # Ensure encoded_word is a Tensor, not a list
#         if not isinstance(encoded_word, torch.Tensor):
#             # Add appropriate dtype and device if needed
#             encoded_word = torch.tensor(encoded_word)

#         batch_labels.append(encoded_word.unsqueeze(0))
#         batch_full_words.append(full_word)

#     return torch.cat(batch_features), torch.cat(batch_missed_chars), \
#         torch.cat(batch_labels), batch_full_words

# # The rest of your create


# def create_val_loader(val_data, char_frequency, max_word_length):
#     val_samples = [create_validation_samples(
#         [game_data]) for game_data in val_data]
#     flattened_val_samples = [
#         sample for sublist in val_samples for sample in sublist]

#     # Define a lambda function to pass the extra arguments
#     def collate_fn_with_args(batch): return validation_collate_fn(
#         batch, char_frequency, max_word_length)

#     val_loader = DataLoader(flattened_val_samples, batch_size=1,
#                             collate_fn=collate_fn_with_args, shuffle=False)
#     return val_loader

# class HangmanDataset(Dataset):
#     def __init__(self, parquet_file, chunk_size=10000):
#         self.parquet_file = parquet_file
#         self.chunk_size = chunk_size
#         self.total_rows = pd.read_parquet(parquet_file, columns=['game_id']).shape[0]
#         self.current_chunk = None
#         self.chunk_start_idx = 0

#     def _load_chunk(self, idx):
#         chunk_idx = idx // self.chunk_size
#         if self.current_chunk is None or self.chunk_start_idx != chunk_idx * self.chunk_size:
#             self.chunk_start_idx = chunk_idx * self.chunk_size
#             self.current_chunk = pd.read_parquet(
#                 self.parquet_file,
#                 skiprows=range(1, self.chunk_start_idx + 1),
#                 nrows=self.chunk_size
#             )

#     def __len__(self):
#         return self.total_rows

#     def __getitem__(self, idx):
#         if idx < self.chunk_start_idx or idx >= self.chunk_start_idx + self.chunk_size:
#             self._load_chunk(idx)
#         row = self.current_chunk.iloc[idx - self.chunk_start_idx]
#         return {
#             'game_id': row['game_id'],
#             'word': row['word'],
#             'initial_state': row['initial_state'].split(','),
#             'final_state': row['final_state'],
#             'guessed_states': row['guessed_states'].split(','),
#             'guessed_letters': row['guessed_letters'].split(','),
#             'game_state': row['game_state'],
#             'difficulty': row['difficulty'],
#             'outcome': row['outcome'],
#             'word_length': row['word_length'],
#             'won': row['won'] == 'True'
#         }
