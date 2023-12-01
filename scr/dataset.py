import gc
import multiprocessing
import pickle
import random
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset, Sampler

from scr.feature_engineering import *

gc.collect()


class PerformanceBasedSampler(Sampler):
    def __init__(self, data_source, performance_dict, random_sampling_rate=0.1):
        self.data_source = data_source
        self.performance_dict = performance_dict
        self.random_sampling_rate = random_sampling_rate
        self.word_length_indices = self.precompute_indices()
        self.calculate_probs()

    def precompute_indices(self):
        indices_dict = defaultdict(list)
        for idx, (_, _, info) in enumerate(self.data_source):
            word_length = str(info['word_length'])
            indices_dict[word_length].append(idx)
        return indices_dict

    def calculate_probs(self):
        win_rates = {}
        for word_length, stats in self.performance_dict.items():
            word_length_str = str(word_length)
            total_games = stats['wins'] + stats['losses']
            win_rate = (stats['wins'] / total_games) * \
                100 if total_games > 0 else 0
            win_rates[word_length_str] = win_rate

        performance_metric = {wl: 1.0 / (win_rate + 0.01)
                              for wl, win_rate in win_rates.items()}
        total = sum(performance_metric.values())
        self.word_length_probs = np.array(
            [metric / total for metric in performance_metric.values()])
        self.word_lengths = list(performance_metric.keys())

    def __iter__(self):
        all_indices = list(range(len(self.data_source)))
        np.random.shuffle(all_indices)

        for idx in all_indices:
            if random.random() < self.random_sampling_rate or not self.word_lengths:
                yield idx
            else:
                word_length = np.random.choice(
                    self.word_lengths, p=self.word_length_probs)
                if word_length in self.word_length_indices and self.word_length_indices[word_length]:
                    selected_index = self.word_length_indices[word_length].pop(
                    )
                    yield selected_index
                else:
                    yield idx

    def __len__(self):
        return len(self.data_source)


class ProcessedHangmanDataset(Dataset):
    def __init__(self, pkls_dir, char_freq, max_word_length, files_limit=None):
        self.char_frequency = char_freq
        self.max_word_length = max_word_length
        # Total number of unique characters
        self.total_char_count = len(self.char_frequency)
        self.data = []

        files_processed = 0
        for batch_dir in sorted(Path(pkls_dir).iterdir(), key=lambda x: int(x.name)
                                if x.name.isdigit() else float('inf')):
            if batch_dir.is_dir():
                for pkl_file in batch_dir.glob("*.pkl"):
                    if files_limit and files_processed >= files_limit:
                        break

                    with open(pkl_file, 'rb') as file:
                        game_states = pickle.load(file)
                        parts = pkl_file.stem.split('-from-')
                        word, remaining = parts[0], parts[1]
                        remaining_parts = remaining.split('-')
                        initial_state = remaining_parts[0]
                        state_name = remaining_parts[1]
                        difficulty = remaining_parts[2]
                        outcome = remaining_parts[3]
                        word_length = remaining_parts[4]

                        for game_state in game_states:
                            game_won, guesses = game_state
                            if len(guesses) > 0:
                                states, next_guesses = self.process_game_states(
                                    guesses)
                                additional_info = {'word': word, 'initial_state': initial_state,
                                                   'game_state': state_name, 'difficulty': difficulty,
                                                   'outcome': outcome, 'word_length': word_length}
                                self.data.append(
                                    (states, next_guesses, additional_info))

                    files_processed += 1
            if files_limit and files_processed >= files_limit:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        game_states, labels, additional_info = self.data[idx]
        return game_states, labels, additional_info

    def process_game_states(self, guesses):
        states = [guesses[0][1]]
        next_guesses = []
        for i in range(1, len(guesses)):
            next_guesses.append(guesses[i][0])
            states.append(guesses[i][1])

        return states[:-1], next_guesses

    def get_word_length_distribution(self):
        word_length_dict = defaultdict(list)
        for idx, (_, _, info) in enumerate(self.data):
            word_length = info['word_length']
            word_length_dict[word_length].append(idx)
        return word_length_dict

    def custom_collate_fn(self, batch):
        # print(f"Received batch (size {len(batch)}): {batch}")  # Debug line

        batch_features, batch_missed_chars, \
            batch_labels, batch_lengths, batch_additional_info = [], [], [], [], []

        # Debug line to check if the batch contains any game_states
        if all(not game_states for game_states, _, _ in batch):
            # print("All game_states in the batch are empty")  # Debug line
            return None, None, None, None, None

        # Calculating the max sequence length with debug line
        game_states_lengths = [len(game_states)
                               for game_states, _, _ in batch if game_states]
        # print(f"game_states lengths: {game_states_lengths}")  # Debug line
        max_seq_length = max(game_states_lengths)

        # Preallocate padding tensors
        padding_tensor_features = torch.zeros(
            (1, self.max_word_length * len(self.char_frequency)))
        padding_tensor_missed_chars = torch.zeros(
            (1, len(self.char_frequency)))

        for item in batch:
            game_states, labels, additional_info = item

            # Debug line to check each item in the batch
            if not game_states:
                # print(f"Encountered an item with empty game_states: {item}")  # Debug line
                continue

            game_features, game_missed_chars = process_game_sequence(
                game_states, self.char_frequency, self.max_word_length, len(game_states))

            original_length = len(game_states)
            batch_lengths.append(original_length)
            batch_additional_info.append(additional_info)

            # Debug line for padding information
            # print(f"Original length: {original_length}, Max sequence length: {max_seq_length}")  # Debug line

            if original_length < max_seq_length:
                padding_length = max_seq_length - original_length

                # Resize padding tensors if necessary
                if padding_tensor_features.shape[1] != game_features.shape[1]:
                    padding_tensor_features = torch.zeros(
                        (1, game_features.shape[1]))
                if padding_tensor_missed_chars.shape[1] != game_missed_chars.shape[1]:
                    padding_tensor_missed_chars = torch.zeros(
                        (1, game_missed_chars.shape[1]))

                # Concatenate features and padding
                game_features_padded = torch.cat(
                    [game_features, padding_tensor_features.repeat(padding_length, 1)], dim=0)
                game_missed_chars_padded = torch.cat(
                    [game_missed_chars, padding_tensor_missed_chars.repeat(padding_length, 1)], dim=0)
            else:
                game_features_padded = game_features
                game_missed_chars_padded = game_missed_chars

            batch_features.append(game_features_padded)
            batch_missed_chars.append(game_missed_chars_padded)
            batch_labels.extend([char_to_idx[label] for label in labels])

        if not batch_features:
            # print("The processed batch is empty after filtering empty game_states.")  # Debug line
            return None, None, None, None, None

        batch_features_stacked = torch.stack(batch_features)
        batch_missed_chars_stacked = torch.stack(batch_missed_chars)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
        lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long)

        return batch_features_stacked, lengths_tensor, \
            batch_missed_chars_stacked, labels_tensor, batch_additional_info

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


def plot_distribution(data, title, ax):
    """
    Helper function to plot the distribution of a specific attribute.
    """
    counter = Counter(data)
    labels, values = zip(*counter.items())
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel('Count')


def analyze_dataset_sanity(dataset, batch_size):
    """
    Analyzes the sanity of the given dataset by plotting the distribution
    of various attributes in a batch of data.
    """
    # Randomly select a batch of data
    batch_data = random.sample(
        dataset.data, min(batch_size, len(dataset.data)))

    # Prepare data for plotting
    word_lengths = [info['word_length'] for _, _, info in batch_data]
    difficulties = [info['difficulty'] for _, _, info in batch_data]
    outcomes = [info['outcome'] for _, _, info in batch_data]
    state_names = [info['game_state'] for _, _, info in batch_data]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plot_distribution(word_lengths, 'Word Length Distribution', axs[0, 0])
    plot_distribution(difficulties, 'Difficulty Distribution', axs[0, 1])
    plot_distribution(outcomes, 'Outcome Distribution', axs[1, 0])
    plot_distribution(state_names, 'State Name Distribution', axs[1, 1])

    plt.tight_layout()
    plt.show()


def analyze_word_length_balance(dataset, batch_size):
    """
    Analyzes the balance of the dataset across different word lengths.
    Each word length category is analyzed individually.
    """
    # Group data by word length
    word_length_groups = defaultdict(list)
    for states, next_guesses, info in dataset.data:
        word_length_groups[info['word_length']].append(
            (states, next_guesses, info))

    # For each word length, analyze the balance
    for word_length, group in word_length_groups.items():
        # Randomly select a batch of data for each word length
        batch_data = random.sample(group, min(batch_size, len(group)))

        # Prepare data for plotting
        difficulties = [info['difficulty'] for _, _, info in batch_data]
        outcomes = [info['outcome'] for _, _, info in batch_data]
        state_names = [info['game_state'] for _, _, info in batch_data]

        # Create subplots for each word length
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Word Length: {word_length}')
        plot_distribution(difficulties, 'Difficulty Distribution', axs[0])
        plot_distribution(outcomes, 'Outcome Distribution', axs[1])
        plot_distribution(state_names, 'State Name Distribution', axs[2])

        plt.tight_layout()
        plt.show()


# Example usage:
# processed_dataset = ProcessedHangmanDataset(...)
# analyze_dataset_sanity(processed_dataset, batch_size=1000)
