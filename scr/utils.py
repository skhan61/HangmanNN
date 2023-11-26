import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch


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
