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
    def __init__(self, pkls_dir, char_freq, max_word_length):
        self.char_frequency = char_freq  # Store char_freq as an attribute
        self.max_word_length = max_word_length
        self.data = []  # Stores tuples of (game_state, label, additional_info)
        # self.mode = mode

        for batch_dir in sorted(pkls_dir.iterdir(), key=lambda x: int(x.name)
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

        # if self.mode == 'individual':
        return states[:-1], next_guesses
        # elif self.mode == 'entire_game':
        #     # Modify this if you need different logic
        #     return [states[0]], [states[-1]]
        # else:
        #     raise ValueError("Invalid mode specified")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_states, labels, additional_info = self.data[idx]
        return game_states, labels, additional_info

    def custom_collate_fn(self, batch):
        batch_features, batch_missed_chars, batch_labels, batch_lengths = [], [], [], []
        max_seq_length = 0  # Variable to track the maximum sequence length in the batch

        # First, process each game to find the maximum sequence length
        for item in batch:
            game_states, labels, _ = item
            if not game_states:
                continue
            max_seq_length = max(max_seq_length, len(game_states))

        # Now, process each game again to pad sequences and collect batch data
        for item in batch:
            game_states, labels, _ = item
            if not game_states:
                continue

            game_features, game_missed_chars = process_game_sequence(
                game_states, self.char_frequency, self.max_word_length, len(game_states))

            # Record the original length of each game state sequence
            original_length = len(game_states)
            batch_lengths.append(original_length)

            # Pad each game's features and missed characters to the maximum sequence length
            if original_length < max_seq_length:
                padding_length = max_seq_length - original_length

                # Create padding tensor for game_features
                padding_tensor_features = torch.zeros(
                    padding_length, game_features.shape[1])
                game_features_padded = torch.cat(
                    [game_features, padding_tensor_features], dim=0)

                # Create a separate padding tensor for game_missed_chars
                padding_tensor_missed_chars = \
                    torch.zeros(padding_length, game_missed_chars.shape[1])
                game_missed_chars_padded = \
                    torch.cat(
                        [game_missed_chars, padding_tensor_missed_chars], dim=0)
            else:
                game_features_padded = game_features
                game_missed_chars_padded = game_missed_chars

            batch_features.append(game_features_padded)
            batch_missed_chars.append(game_missed_chars_padded)
            batch_labels.extend([char_to_idx[label] for label in labels])

        # Before stacking, check if the lists are empty
        if not batch_features or not batch_missed_chars:
            # Handle empty batch here, maybe skip or return None
            print("Encountered an empty batch")
            return None, None, None, None

        # Stack all games to form the batch
        batch_features_stacked = torch.stack(batch_features)
        batch_missed_chars_stacked = torch.stack(batch_missed_chars)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
        lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long)

        return batch_features_stacked, lengths_tensor, batch_missed_chars_stacked, labels_tensor

    def create_validation_samples(self, game_data):
        validation_samples = []
        for i in range(len(game_data[0]) - 1):
            current_state = game_data[0][i]
            guessed_character = game_data[1][i]
            next_character = game_data[1][i + 1]
            full_word = game_data[2]['word']
            validation_samples.append(
                ([current_state, guessed_character], next_character, full_word))
        return validation_samples

    def validation_collate_fn(self, batch):
        batch_features, batch_missed_chars, batch_labels, \
            batch_full_words = [], [], [], []

        for item in batch:
            game_state, full_word = item
            processed_state, missed_chars = process_single_game_state(
                game_state, self.char_frequency, self.max_word_length)

            # Append the processed features and missed characters
            processed_state = processed_state.unsqueeze(0)
            missed_chars = missed_chars.unsqueeze(0)
            batch_features.append(processed_state)
            batch_missed_chars.append(missed_chars)

            # Ensure labels are tensors
            encoded_label = encode_word(full_word)
            if not isinstance(encoded_label, torch.Tensor):
                encoded_label = torch.tensor(encoded_label, dtype=torch.long)
            batch_labels.append(encoded_label)

            # Also, append the full word for each item
            batch_full_words.append(full_word)

        # Stacking the batch tensors
        batch_features_tensor = torch.stack(batch_features)
        batch_missed_chars_tensor = torch.stack(batch_missed_chars)
        batch_labels_tensor = torch.stack(batch_labels)

        return batch_features_tensor, batch_missed_chars_tensor, \
            batch_labels_tensor, batch_full_words

    def create_val_loader(self, val_data):
        val_samples = []
        for game_data in val_data:
            # Each game_data is a tuple of (game_states, guesses, additional_info)
            game_states, guesses, additional_info = game_data
            for i in range(len(game_states) - 1):
                current_state = game_states[i]
                next_guess = guesses[i + 1]
                full_word = additional_info['word']
                val_samples.append(([current_state, next_guess], full_word))

        # Use validation_collate_fn to process each sample
        val_loader = DataLoader(val_samples, batch_size=1,
                                collate_fn=self.validation_collate_fn, shuffle=False)
        return val_loader

    # def create_val_loader(self, val_data):
    #     val_samples = [self.create_validation_samples(
    #         data) for data in val_data]
    #     flattened_val_samples = [
    #         sample for sublist in val_samples for sample in sublist]

    #     val_loader = DataLoader(flattened_val_samples, batch_size=1,
    #                             collate_fn=self.validation_collate_fn, shuffle=False)
    #     return val_loader
