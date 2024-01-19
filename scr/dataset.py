import os
from collections import OrderedDict, defaultdict

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *


class HangmanDataset(Dataset):
    def __init__(self, parquet_files):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.word_length_index = defaultdict(list)

        # Build the word length index
        for file_idx, file in enumerate(self.parquet_files):
            df = pd.read_parquet(file)
            for local_idx, row in df.iterrows():
                self.word_length_index[row['word_length']].append(
                    (file_idx, local_idx))

        self.cumulative_lengths = self._compute_cumulative_lengths(
            [len(v) for v in self.word_length_index.values()])

        # print(self.word_length_index)

    def _compute_cumulative_lengths(self, lengths):
        cum_lengths = [0]
        total = 0
        for length in lengths:
            total += length
            cum_lengths.append(total)
        return cum_lengths

    def _get_file_and_local_idx(self, word_length, idx):
        file_idx, local_idx = self.word_length_index[word_length][idx]
        return self.parquet_files[file_idx], local_idx

    def __len__(self):
        return sum(len(indices) for indices in self.word_length_index.values())

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            word_length, idx = idx
        else:
            # Find word length for the given idx
            for word_length, indices in self.word_length_index.items():
                if idx < len(indices):
                    break
                idx -= len(indices)

        file_path, local_idx = self._get_file_and_local_idx(word_length, idx)
        df = pd.read_parquet(file_path)
        row = df.iloc[local_idx]

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
    max_seq_len = max(len(item['guessed_states']) for item in batch)

    # Preallocate arrays with maximum sequence length
    padded_states = [[''] * max_seq_len for _ in batch]
    padded_letters = [[''] * max_seq_len for _ in batch]
    original_seq_lengths = [len(item['guessed_states']) for item in batch]

    for i, item in enumerate(batch):
        seq_len = original_seq_lengths[i]
        padded_states[i][:seq_len] = item['guessed_states']
        padded_letters[i][:seq_len] = item['guessed_letters']

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

# class HangmanDataset(Dataset):
#     def __init__(self, parquet_files):
#         self.parquet_files = parquet_files


# class HangmanDataset(Dataset):
#     def __init__(self, parquet_files):
#         # Ensure parquet_files is a list, even if it's a single path
#         if not isinstance(parquet_files, list):
#             parquet_files = [parquet_files]

#         self.parquet_files = parquet_files
#         self.lengths = [pd.read_parquet(f).shape[0]
#                         for f in self.parquet_files]
#         self.cumulative_lengths = self._compute_cumulative_lengths(
#             self.lengths)

#     def _compute_cumulative_lengths(self, lengths):
#         cum_lengths = [0]
#         total = 0
#         for length in lengths:
#             total += length
#             cum_lengths.append(total)
#         return cum_lengths

#     def _get_file_and_local_idx(self, idx):
#         for file_idx, cum_length in enumerate(self.cumulative_lengths):
#             if idx < cum_length:
#                 local_idx = idx - self.cumulative_lengths[file_idx - 1]
#                 return self.parquet_files[file_idx - 1], local_idx
#         raise IndexError("Index out of bounds")

#     def __len__(self):
#         return self.cumulative_lengths[-1]

#     def __getitem__(self, idx):
#         file_path, local_idx = self._get_file_and_local_idx(idx)
#         df = pd.read_parquet(file_path)
#         row = df.iloc[local_idx]

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
