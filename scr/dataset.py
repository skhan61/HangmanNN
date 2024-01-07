from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import *


class HangmanDataset(Dataset):
    def __init__(self, parquet_file):
        self.dataframe = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
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


def create_validation_samples(game_data_list):
    validation_samples = []
    for game_data in game_data_list:
        for i in range(len(game_data['guessed_states']) - 1):
            current_state = game_data['guessed_states'][i]
            next_guess = game_data['guessed_letters'][i + 1]
            full_word = game_data['word']
            validation_samples.append(([current_state, next_guess], full_word))
    return validation_samples


def validation_collate_fn(batch, char_frequency, max_word_length):

    batch_features, batch_missed_chars, batch_labels, batch_full_words = [], [], [], []

    for game_state, full_word in batch:

        print(f'Game states from valid collate: ', game_state)
        processed_state, missed_chars = process_batch_of_games(game_state,
                                                               guessed_letters_batch,
                                                               char_frequency,
                                                               max_word_length,
                                                               max_seq_length)
        batch_features.append(processed_state.unsqueeze(0))
        batch_missed_chars.append(missed_chars.unsqueeze(0))
        batch_labels.append(encode_word(full_word).unsqueeze(0))
        batch_full_words.append(full_word)

    return torch.cat(batch_features), torch.cat(batch_missed_chars), \
        torch.cat(batch_labels), batch_full_words


def create_val_loader(val_data, char_frequency, max_word_length):
    val_samples = [create_validation_samples(
        [game_data]) for game_data in val_data]
    flattened_val_samples = [
        sample for sublist in val_samples for sample in sublist]

    # Define a lambda function to pass the extra arguments
    def collate_fn_with_args(batch): return validation_collate_fn(
        batch, char_frequency, max_word_length)

    val_loader = DataLoader(flattened_val_samples, batch_size=1,
                            collate_fn=collate_fn_with_args, shuffle=False)
    return val_loader
