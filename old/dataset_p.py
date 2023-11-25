import gc
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *

gc.collect()


def process_pkl_file(pkl_file_path, char_freq_dict, max_word_length, mode):
    data = []
    with open(pkl_file_path, 'rb') as file:
        game_states = pickle.load(file)

        parts = Path(pkl_file_path).stem.split('_from_')
        word, remaining = parts[0], parts[1].split('_')
        initial_state, difficulty, outcome = '_'.join(
            remaining[:-2]), remaining[-2], remaining[-1]

        for game_state in game_states:
            game_won, guesses = game_state
            if len(guesses) > 0:
                states, next_guesses = process_game_states(guesses, mode)
                additional_info = {'word': word, 'initial_state': initial_state,
                                   'difficulty': difficulty, 'outcome': outcome}
                data.append((states, next_guesses, additional_info))
    return data


def process_game_states(guesses, mode):
    states = [guesses[0][1]]
    next_guesses = []
    for i in range(1, len(guesses)):
        next_guesses.append(guesses[i][0])
        states.append(guesses[i][1])

    if mode == 'individual':
        return states[:-1], next_guesses
    elif mode == 'entire_game':
        return [states[0]], [states[-1]]
    else:
        raise ValueError("Invalid mode specified")


class ProcessedHangmanDataset(Dataset):
    def __init__(self, pkls_dir, char_freq_dict,
                 max_word_length, mode='individual'):
        self.char_freq_dict = char_freq_dict
        self.max_word_length = max_word_length
        self.data = []  # Stores tuples of (game_state, label, additional_info)
        self.mode = mode

        # Prepare a list of all .pkl files
        pkl_files = [str(pkl_file) for batch_dir in sorted(Path(pkls_dir).iterdir(),
                                                           key=lambda x: int(
                                            x.name)
                                            if x.name.isdigit() else float('inf'))
                                            if batch_dir.is_dir() for pkl_file in batch_dir.glob("*.pkl")]

        # Use multiprocessing Pool to process files in parallel
        with Pool() as pool:
            results = pool.starmap(process_pkl_file,
                                   [(pkl_file, self.char_freq_dict, self.max_word_length,
                                    self.mode) for pkl_file in pkl_files])

        # Combine results from all files
        for result in results:
            self.data.extend(result)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_states, labels, additional_info = self.data[idx]
        return game_states, labels, additional_info


# class ProcessedHangmanDataset(Dataset):
#     def __init__(self, pkls_dir, char_freq_dict, max_word_length, mode='individual'):
#         self.char_freq_dict = char_freq_dict
#         self.max_word_length = max_word_length
#         self.data = []  # Stores tuples of (game_state, label, additional_info)
#         self.mode = mode

#         for batch_dir in sorted(pkls_dir.iterdir(), key=lambda x: int(x.name)
#                                 if x.name.isdigit() else float('inf')):
#             if batch_dir.is_dir():
#                 for pkl_file in batch_dir.glob("*.pkl"):
#                     with open(pkl_file, 'rb') as file:
#                         game_states = pickle.load(file)

#                         parts = pkl_file.stem.split('_from_')
#                         word, remaining = parts[0], parts[1].split('_')
#                         initial_state, difficulty, outcome = '_'.join(
#                             remaining[:-2]), remaining[-2], remaining[-1]

#                         for game_state in game_states:
#                             game_won, guesses = game_state
#                             if len(guesses) > 0:
#                                 states, next_guesses = self.process_game_states(
#                                     guesses)
#                                 additional_info = {'word': word, 'initial_state': initial_state,
#                                                    'difficulty': difficulty, 'outcome': outcome}
#                                 self.data.append(
#                                     (states, next_guesses, additional_info))

#     def process_game_states(self, guesses):
#         states = [guesses[0][1]]
#         next_guesses = []
#         for i in range(1, len(guesses)):
#             next_guesses.append(guesses[i][0])
#             states.append(guesses[i][1])

#         if self.mode == 'individual':
#             return states[:-1], next_guesses
#         elif self.mode == 'entire_game':
#             # Modify this if you need different logic
#             return [states[0]], [states[-1]]
#         else:
#             raise ValueError("Invalid mode specified")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         game_states, labels, additional_info = self.data[idx]
#         return game_states, labels, additional_info
