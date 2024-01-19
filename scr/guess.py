import random
import re
import string
from collections import Counter
from itertools import combinations

import torch
import torch.nn as nn

# process_single_word, get_missed_characters, char_to_idx, idx_to_char
from scr.feature_engineering import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_random_character():
#     # Choose from lowercase letters
#     characters = 'a'  # string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
#     random_char = random.choice(characters)
#     return random_char


# # # def gues(self, word):
def guess_character(model, masked_word, char_frequency,
                    max_word_length, guessed_chars,
                    max_seq_length=1, fallback_strategy=True, device=None):
    """
    Guess the next character in the hangman game.

    Args:
        model: Trained RNN model. The neural network model used for character prediction.
        masked_word (str): Current state of the word being guessed, represented with '_' 
        for missing characters. char_frequency (dict): Frequency of each character in the 
        training set, used to determine the likelihood of each character.
        max_word_length (int): Maximum length of words in the training set. This is used to 
        normalize word lengths in the model. device: The device (CPU/GPU) on which the model is 
        running, ensuring compatibility and performance optimization.
        guessed_chars (set): Set of characters that have already been guessed in the current game.
        fallback_strategy (bool): Flag to determine whether to use a fallback strategy or not. 
        The fallback strategy is used when the model's prediction is uncertain.

    Returns:
        str: The character guessed by the model or the fallback strategy.
    """
    fets, missed_chars = process_batch_of_games([masked_word],
                                                char_frequency, max_word_length, max_seq_length=1)

    # print(f"fets shape form guess character: ", fets.shape)
    # print(f"missed character shape: ", missed_chars.shape)

    seq_lens = torch.tensor(
        [fets.size(1)], dtype=torch.long)  # , device=self.device)

    # fets, missed_chars = fets.to(self.device), missed_chars.to(
    #     self.device)  # Move tensors to correct device

    model.eval()

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    fets = fets.to(device)
    missed_chars = missed_chars.to(device)

    with torch.no_grad():
        output = model(fets, seq_lens, missed_chars)

    # print(f"Output shape: {output.shape}")

    last_char_position = seq_lens.item() - 1
    probabilities = torch.sigmoid(output[0, last_char_position, :])
    # probabilities = torch.softmax(output[0, last_char_position, :])

    # Exclude already guessed characters
    guessed_indices = [char_to_idx[char]
                       for char in guessed_chars if char in char_to_idx]

    probabilities[torch.tensor(
        guessed_indices, dtype=torch.long, device=device)] = 0

    best_char_index = torch.argmax(probabilities).item()
    # print(best_char_index)
    guessed_char = idx_to_char[best_char_index]

    # Fallback strategy
    if fallback_strategy and (guessed_char in guessed_chars or guessed_char == '_'):
        for char, _ in sorted(char_frequency.items(), key=lambda x: x[1], reverse=True):
            if char not in guessed_chars:
                guessed_char = char
                break

    return guessed_char

# def gues(self, word):


def guess(model, word, char_frequency,
          max_word_length, guessed_letters):

    # word == state here

    cleaned_word = "".join(char.lower()
                           for char in word if char.isalpha() or char == '_')
    # print(cleaned_word)

    # Predict the next character using the updated guess_character function
    guessed_char = guess_character(
        model, cleaned_word,
        char_frequency,
        max_word_length,
        guessed_letters  # Pass the list of guessed letters
    )

    # guessed_char =  get_random_character()

    # Add the new guess to the guessed letters list
    if guessed_char not in guessed_letters:
        guessed_letters.append(guessed_char)

    return guessed_char
