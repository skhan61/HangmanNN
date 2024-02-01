import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *


class HangmanDataset(Dataset):
    def __init__(self, parquet_path_or_files, indices=None):
        # Handle both a single path (str or Path) and a list of Path objects
        if isinstance(parquet_path_or_files, list):
            self.parquet_files = [Path(file) if not isinstance(
                file, Path) else file for file in parquet_path_or_files]
        elif Path(parquet_path_or_files).is_dir():
            self.parquet_files = list(
                Path(parquet_path_or_files).glob('*.parquet'))
        else:
            self.parquet_files = [Path(parquet_path_or_files)]

        # print(len(self.parquet_files))
        self.seq_len_index = defaultdict(list)
        self.total_length = 0
        if indices is not None:
            self.seq_len_index = indices
            self.total_length = sum(len(v)
                                    for v in self.seq_len_index.values())
        else:
            self.build_seq_len_index()

    def build_seq_len_index(self):
        for file_idx, file in enumerate(self.parquet_files):
            df = pd.read_parquet(file)
            for local_idx, row in df.iterrows():
                seq_len_key = len(row['guessed_states'].split(','))
                self.seq_len_index[seq_len_key].append((file_idx, local_idx))
        self.total_length = sum(len(indices)
                                for indices in self.seq_len_index.values())

    def rebuild_seq_len_index(self):
        self.seq_len_index = defaultdict(list)  # Reset the index
        self.total_length = 0  # Reset the total length
        self.build_seq_len_index()  # Rebuild the index using the existing method

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 2:
            file_idx, local_idx = idx
        elif isinstance(idx, int):
            if 0 <= idx < self.total_length:
                for seq_len, indices in self.seq_len_index.items():
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
            'won': row['won'] # == 'True'
        }

    def split(self, test_size=0.2):
        train_indices = defaultdict(list)
        valid_indices = defaultdict(list)

        for seq_len, indices in self.seq_len_index.items():
            train_idx, valid_idx = train_test_split(
                indices, test_size=test_size, random_state=42)
            train_indices[seq_len].extend(train_idx)
            valid_indices[seq_len].extend(valid_idx)

        train_dataset = HangmanDataset(
            self.parquet_files, indices=train_indices)
        valid_dataset = HangmanDataset(
            self.parquet_files, indices=valid_indices)

        return train_dataset, valid_dataset

    def flat_index_to_tuple(self, flat_index):
        cumulative_count = 0
        for seq_len, indices in self.seq_len_index.items():
            if cumulative_count + len(indices) > flat_index:
                local_index = flat_index - cumulative_count
                # Assumes indices are stored as (file_index, data_index) tuples
                return indices[local_index]
            cumulative_count += len(indices)
        raise IndexError("Flat index out of range.")


# class HangmanDataset(Datase):
#     def __init__(self, parquet_files):
#         self.parquet_files = parquet_files if isinstance(
#             parquet_files, list) else [parquet_files]
#         self.pair_index = defaultdict(list)
#         self.word_length_index = defaultdict(list)
#         self.seq_len_index = defaultdict(list)
#         self.total_length = 0
#         self.build_pair_index()

#     def build_pair_index(self):
#         for file_idx, file in enumerate(self.parquet_files):
#             df = pd.read_parquet(file)
#             for local_idx, row in df.iterrows():
#                 difficulty_outcome_key = (row['difficulty'], row['outcome'])
#                 word_length_key = row['word_length']
#                 seq_len_key = len(row['guessed_states'].split(','))

#                 self.pair_index[difficulty_outcome_key].append(
#                     (file_idx, local_idx))
#                 self.word_length_index[word_length_key].append(
#                     (file_idx, local_idx))
#                 self.seq_len_index[seq_len_key].append((file_idx, local_idx))

#         self.total_length = sum(len(indices)
#                                 for indices in self.pair_index.values())

#     def rebuild_pair_index(self, target_pairs):
#         # Reinitialize indices
#         self.pair_index = defaultdict(list)
#         self.word_length_index = defaultdict(list)
#         self.seq_len_index = defaultdict(list)

#         # # Debug: Print the start of index rebuilding
#         # print("Rebuilding indices...")

#         # Rebuild the indices based on target pairs
#         for file_idx, file in enumerate(self.parquet_files):
#             df = pd.read_parquet(file)
#             for local_idx, row in df.iterrows():
#                 difficulty_outcome_key = (row['difficulty'], row['outcome'])
#                 word_length_key = row['word_length']
#                 seq_len_key = len(row['guessed_states'].split(','))

#                 # Check if the current row matches any of the target pairs
#                 if any(word_length_key == pair[1] for pair in target_pairs if pair[0] == 'word_len'):
#                     self.word_length_index[word_length_key].append(
#                         (file_idx, local_idx))
#                     # print(
#                     #     f"Indexing word length {word_length_key} at ({file_idx}, {local_idx})")

#         # Update the total length of the dataset
#         self.total_length = sum(len(indices)
#                                 for indices in self.pair_index.values())

#         # # Debug: Print completion of index rebuilding
#         # print("Index rebuilding complete.")

#     def __len__(self):
#         return self.total_length

#     def __getitem__(self, idx):
#         if isinstance(idx, tuple) and len(idx) == 4:
#             # Tuple-based indexing for difficulty, outcome, file index, and local index
#             _, _, file_idx, local_idx = idx
#         elif isinstance(idx, tuple) and len(idx) == 2:
#             length_type, length_value = idx
#             if length_type == 'seq_len':
#                 index = self.seq_len_index
#             elif length_type == 'word_len':
#                 index = self.word_length_index
#             else:
#                 raise ValueError(
#                     "Invalid length type. Must be 'seq_len' or 'word_len'.")

#             if length_value in index and index[length_value]:
#                 file_idx, local_idx = random.choice(index[length_value])
#             else:
#                 raise IndexError(
#                     f"No data available for the given {length_type}.")
#         elif isinstance(idx, int):
#             if 0 <= idx < self.total_length:
#                 for indices in self.pair_index.values():
#                     if idx < len(indices):
#                         file_idx, local_idx = indices[idx]
#                         break
#                     idx -= len(indices)
#             else:
#                 raise IndexError("Index out of range.")
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
#         return {key: len(indices) for key, indices in self.pair_index.items()}

#     def get_all_group_labels(self):
#         group_labels = set()
#         word_lengths = set()
#         for file in self.parquet_files:
#             df = pd.read_parquet(file)
#             group_labels.update((row['difficulty'], row['outcome'])
#                                 for row in df.itertuples())
#             word_lengths.update(df['word_length'])
#         return list(group_labels), list(word_lengths)


def new_custom_collate_fn(batch):

    # print(f"Batch size before processing: {len(batch)}")  # Debugging print

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
    game_ids = [item['game_id'] for item in batch]
    initial_states = [item['initial_state'] for item in batch]
    final_states = [item['final_state'] for item in batch]
    game_states = [item['game_state'] for item in batch]

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
        'won': won_flags,
        'initial_state': initial_states,
        'final_state': final_states,
        'game_state': game_states
    }


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
