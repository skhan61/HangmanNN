import collections
import pickle
import random
import re
from collections import defaultdict
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset, Subset

from scr.dataset import *


def determine_current_state(masked_word, guessed_chars):
    word_length = len(masked_word)
    num_revealed_chars = sum(1 for char in masked_word if char != '_')
    num_guessed_unique_chars = len(set(guessed_chars))

    # Estimate the state based on the number of revealed characters
    if num_revealed_chars == 0:
        return "allMasked"
    elif num_revealed_chars <= 1 and word_length > 4:
        return "early"
    elif num_revealed_chars <= num_guessed_unique_chars // 4:
        return "quarterRevealed"
    elif num_revealed_chars <= num_guessed_unique_chars // 2:
        return "midRevealed"
    elif num_revealed_chars <= 3 * num_guessed_unique_chars // 4:
        return "midLateRevealed"
    elif num_revealed_chars < num_guessed_unique_chars:
        return "lateRevealed"
    else:
        return "nearEnd"


def calculate_difficulty_score(metrics, weight_win_rate=1.0,
                               weight_miss_penalty=0.5):
    """

    Calculates the difficulty score based on win rate and miss penalty.

    :param metrics: Dictionary containing 'performance_wins' and 'miss_penalty_avg'.
    :param weight_win_rate: Weight for the win rate metric.
    :param weight_miss_penalty: Weight for the miss penalty metric.
    :return: Calculated difficulty score.

    Best Case Scenario (Lowest Difficulty for Players):
    - Win Rate: 100% - The word is guessed correctly almost every time, 
        indicating it is easy for players.
    - Miss Penalty: 0 - There are rarely any incorrect guesses for this word.
    - Difficulty Score: 0 - This score suggests that the word is the least 
        challenging for players.

    Worst Case Scenario (Highest Difficulty for Players):
    - Win Rate: 0% - The word is almost never guessed correctly, 
        indicating it is very difficult for players.
    - Miss Penalty: 1 - There is a high frequency of 
        incorrect guesses for this word.
    - Difficulty Score: 1.5 - This score suggests 
        that the word is highly challenging for players.

    """
    # Extracting the metrics
    win_rate = metrics.get('performance_wins', 0)
    miss_penalty = metrics.get('miss_penalty_avg', 0)

    # Normalize the metrics (invert win rate as lower win rate indicates higher difficulty)
    normalized_win_rate = (100 - win_rate) / 100
    normalized_miss_penalty = miss_penalty  # Already in range 0 to 1

    # Calculate the composite score
    composite_score = (
        weight_win_rate * normalized_win_rate +
        weight_miss_penalty * normalized_miss_penalty
    )

    return composite_score


# Function to flatten a nested dictionary
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reorganize_and_aggregate_metrics(flattened_data):
    seq_len_stats = {}

    # Regular expression to match sequence length-related miss penalty stats
    seq_len_pattern = re.compile(r'seq_length_(\d+)_miss_penalty')

    for key, value in flattened_data.items():
        seq_len_match = seq_len_pattern.match(key)

        if seq_len_match:
            # Extract and convert sequence length to integer
            seq_length = int(seq_len_match.group(1))

            # Aggregate miss penalty statistics based on sequence length
            # Assign miss penalty value directly
            seq_len_stats[seq_length] = value

    return seq_len_stats


def flatten_for_logging(aggregated_metrics):
    loggable_metrics = {}
    for word_len, stats in aggregated_metrics.items():
        for stat, value in stats.items():
            flattened_key = f'{stat}_{word_len}'
            loggable_metrics[flattened_key] = value
    return loggable_metrics


def save_dataset_pickle(dataset, file_name='data/train_dataset.pkl'):
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset_pickle(file_name='data/train_dataset.pkl'):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():  # PyTorch CUDA
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    pd.options.compute.use_bottleneck = False
    pd.options.compute.use_numexpr = False

# # Usage example:
# set_seed(42)


def print_scenarios(scenarios):
    for scenario in scenarios:
        # Print all key-value pairs except 'data'
        for key, value in scenario.items():
            if key != 'data':
                print(f"{key.capitalize()}: {value}")

        # Special handling for 'data' key
        if 'data' in scenario:
            game_won, guesses = scenario['data']
            print(f"  Game {'Won' if game_won else 'Lost'}")
            for guess in guesses:
                letter, state, correct = guess
                print(
                    f"  Guessed '{letter}', State: {state}, Correct: {correct}")
        print("")

# Read a subset of words for debugging


def read_words(filepath, limit=None):
    with open(filepath, 'r') as file:
        words = [line.strip() for line in file]
        if limit:
            words = words[:limit]
    return words

# Function to save a list of words to a file


def save_words_to_file(word_list, file_path):
    with open(file_path, 'w') as file:
        for word in word_list:
            file.write(word + '\n')


def sample_words(words, total_sample_size=1000):
    # Group words by length
    words_by_length = {}
    for word in words:
        length = len(word)
        words_by_length.setdefault(length, []).append(word)

    # Sample words
    sampled_words = []
    lengths = list(words_by_length.keys())
    random.shuffle(lengths)

    for length in lengths:
        num_words_needed = total_sample_size - len(sampled_words)
        words_to_sample = min(len(words_by_length[length]), num_words_needed)
        sampled_words.extend(random.sample(
            words_by_length[length], words_to_sample))
        if len(sampled_words) >= total_sample_size:
            break

    return sampled_words


# def flatten_dict(d, parent_key='', sep='_'):
#     items = []
#     for k, v in d.items():
#         new_key = f'{parent_key}{sep}{k}' if parent_key else k
#         if isinstance(v, collections.abc.MutableMapping):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)


def plot_hangman_stats(data):
    # Prepare data for plotting
    word_lengths = []
    win_rates = []
    average_attempts = []
    total_games = []

    for length, stats in data.items():
        word_lengths.append(length)
        win_rates.append(stats['win_rate'])
        average_attempts.append(stats['average_attempts_used'])
        total_games.append(stats['total_games'])

    # Plotting
    plt.figure(figsize=(18, 6))

    # Win Rate by Word Length
    plt.subplot(1, 3, 1)
    sns.barplot(x=word_lengths, y=win_rates, palette="coolwarm")
    plt.title("Win Rate by Word Length", fontsize=16)
    plt.xlabel("Word Length", fontsize=14)
    plt.ylabel("Win Rate (%)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Average Attempts Used by Word Length
    plt.subplot(1, 3, 2)
    sns.barplot(x=word_lengths, y=average_attempts, palette="viridis")
    plt.title("Average Attempts by Word Length", fontsize=16)
    plt.xlabel("Word Length", fontsize=14)
    plt.ylabel("Average Attempts", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Total Games per Word Length
    plt.subplot(1, 3, 3)
    sns.barplot(x=word_lengths, y=total_games, palette="magma")
    plt.title("Total Games per Word Length", fontsize=16)
    plt.xlabel("Word Length", fontsize=14)
    plt.ylabel("Total Games", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

