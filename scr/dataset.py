import os
import random
from collections import OrderedDict, defaultdict

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *


class HangmanDataset(Dataset):
    def __init__(self, parquet_files):
        self.parquet_files = parquet_files if isinstance(
            parquet_files, list) else [parquet_files]
        self.pair_index = defaultdict(list)
        self.word_length_index = defaultdict(list)  # New attribute
        self.total_length = 0
        self.build_pair_index()

    def build_pair_index(self):
        self.word_length_index = defaultdict(
            list)  # Initialize word length index

        for file_idx, file in enumerate(self.parquet_files):
            df = pd.read_parquet(file)
            for local_idx, row in df.iterrows():
                difficulty_outcome_key = (row['difficulty'], row['outcome'])
                word_length_key = row['word_length']
                self.pair_index[difficulty_outcome_key].append(
                    (file_idx, local_idx))
                self.word_length_index[word_length_key].append(
                    (file_idx, local_idx))

        self.total_length = sum(len(indices)
                                for indices in self.pair_index.values())

    def rebuild_pair_index(self, target_pairs):
        new_pair_index = defaultdict(list)
        new_word_length_index = defaultdict(list)  # New word length index

        for file_idx, file in enumerate(self.parquet_files):
            df = pd.read_parquet(file)
            for local_idx, row in df.iterrows():
                difficulty_outcome_key = (row['difficulty'], row['outcome'])
                word_length_key = row['word_length']
                if difficulty_outcome_key in target_pairs:
                    new_pair_index[difficulty_outcome_key].append(
                        (file_idx, local_idx))
                    new_word_length_index[word_length_key].append(
                        (file_idx, local_idx))

        self.pair_index = new_pair_index
        self.word_length_index = new_word_length_index  # Update the word length index
        self.total_length = sum(len(indices)
                                for indices in new_pair_index.values())

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        if isinstance(idx, tuple) and len(idx) == 4:
            # Tuple-based indexing for difficulty, outcome, file index, and local index
            difficulty, outcome, file_idx, local_idx = idx

        elif isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
            # Tuple-based indexing for word length and index
            word_length = idx[0]
            if word_length in self.word_length_index and self.word_length_index[word_length]:
                file_idx, local_idx = random.choice(
                    self.word_length_index[word_length])
            else:
                raise IndexError(
                    "No data available for the given word length.")

        elif isinstance(idx, int):
            # Integer-based indexing; retrieve a random sample from the entire dataset
            if self.total_length > 0 and 0 <= idx < self.total_length:
                for _, indices in self.pair_index.items():
                    if idx < len(indices):
                        file_idx, local_idx = indices[idx]
                        break
                    idx -= len(indices)
            else:
                raise IndexError("Index out of range.")
        else:
            raise ValueError(
                "Invalid index type. Must be a tuple or an integer.")

        file_path = self.parquet_files[file_idx]
        df = pd.read_parquet(file_path)
        row = df.iloc[local_idx]
        return self.row_to_dict(row)

    def row_to_dict(self, row):
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

    def get_group_info(self):
        """
        Returns the number of samples for each (difficulty, outcome) pair.
        """
        return {key: len(indices) for key, indices in self.pair_index.items()}

    def get_all_group_labels(self):
        """
        Returns a list of all unique group labels (difficulty, outcome) and unique word lengths in the dataset.
        """
        group_labels = set()
        word_lengths = set()
        for file in self.parquet_files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                group_label = (row['difficulty'], row['outcome'])
                group_labels.add(group_label)
                word_lengths.add(row['word_length'])
        return list(group_labels), list(word_lengths)

# ================================================
# class HangmanDataset(Dataset):
#     def __init__(self, parquet_files):
#         self.parquet_files = parquet_files if isinstance(
#             parquet_files, list) else [parquet_files]
#         self.pair_index = defaultdict(list)
#         self.total_length = 0
#         self.build_pair_index()

#     def build_pair_index(self):
#         for file_idx, file in enumerate(self.parquet_files):
#             df = pd.read_parquet(file)
#             grouped = df.groupby(['difficulty', 'outcome']).apply(
#                 lambda x: list(x.index))
#             for key, indices in grouped.items():
#                 self.pair_index[key].extend(
#                     [(file_idx, local_idx) for local_idx in indices])
#         self.total_length = sum(len(indices)
#                                 for indices in self.pair_index.values())

#     def rebuild_pair_index(self, target_pairs):
#         """
#         Rebuilds the pair index based on the specified target pairs.
#         """
#         new_pair_index = defaultdict(list)
#         for file_idx, file in enumerate(self.parquet_files):
#             df = pd.read_parquet(file)
#             for local_idx, row in df.iterrows():
#                 key = (row['difficulty'], row['outcome'])
#                 if key in target_pairs:
#                     new_pair_index[key].append((file_idx, local_idx))

#         self.pair_index = new_pair_index
#         self.total_length = sum(len(indices)
#                                 for indices in self.pair_index.values())

#     def __len__(self):
#         return self.total_length

#     def __getitem__(self, idx):
#         if isinstance(idx, tuple):
#             difficulty, outcome, file_idx, local_idx = idx
#         elif isinstance(idx, int):
#             idx_copy = idx
#             for key, indices in self.pair_index.items():
#                 if idx_copy < len(indices):
#                     file_idx, local_idx = indices[idx_copy]
#                     break
#                 idx_copy -= len(indices)
#         else:
#             raise ValueError(
#                 "Invalid index type. Must be a tuple or an integer.")

#         file_path = self.parquet_files[file_idx]
#         df = pd.read_parquet(file_path)
#         row = df.iloc[local_idx]
#         return self.row_to_dict(row)

#     def row_to_dict(self, row):
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

#     def get_group_info(self):
#         """
#         Returns the number of samples for each (difficulty, outcome) pair.
#         """
#         return {key: len(indices) for key, indices in self.pair_index.items()}

#     def get_all_group_labels(self):
#         """
#         Returns a list of all group labels (word_length, difficulty, outcome) in the dataset.
#         """
#         group_labels = []
#         for file in self.parquet_files:
#             df = pd.read_parquet(file)
#             for _, row in df.iterrows():
#                 group_label = (row['difficulty'], row['outcome'])
#                 group_labels.append(group_label)
#         return group_labels

#     def get_group_info(self):
#         return {key: len(indices) for key, indices in self.pair_index.items()}


def custom_collate_fn(batch):
    max_seq_len = max(len(item['guessed_states']) for item in batch)

    # Preallocate arrays with maximum sequence length
    padded_states = [[''] * max_seq_len for _ in batch]
    padded_letters = [[''] * max_seq_len for _ in batch]
    original_seq_lengths = [len(item['guessed_states']) for item in batch]

    # Additional fields
    difficulties = [item['difficulty'] for item in batch]
    outcomes = [item['outcome'] for item in batch]
    word_lengths = [item['word_length'] for item in batch]
    won_flags = [item['won'] for item in batch]

    for i, item in enumerate(batch):
        seq_len = original_seq_lengths[i]
        padded_states[i][:seq_len] = item['guessed_states']
        padded_letters[i][:seq_len] = item['guessed_letters']

    return {
        'guessed_states': padded_states,
        'guessed_letters': padded_letters,
        'max_seq_len': max_seq_len,
        'original_seq_lengths': original_seq_lengths,
        'difficulty': difficulties,
        'outcome': outcomes,
        'word_length': word_lengths,
        'won': won_flags
    }
