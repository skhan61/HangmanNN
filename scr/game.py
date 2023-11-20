import torch
import torch.nn as nn
import random
from collections import Counter
from scr.feature_engineering import * # process_single_word, get_missed_characters, char_to_idx, idx_to_char

from scr.feature_engineering import process_single_word_inference, \
    char_to_idx, idx_to_char

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import re
from collections import Counter
import random
import string

def get_random_character():
    # Choose from lowercase letters
    characters = 'a' # string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
    random_char = random.choice(characters)
    return random_char


# # def gues(self, word):
def guess_character(model, masked_word, char_frequency, \
    max_word_length, device, guessed_chars, fallback_strategy=True):
    """
    Guess the next character in the hangman game.

    :param model: Trained RNN model.
    :param masked_word: Current state of the word being guessed with '_' for missing characters.
    :param char_frequency: Frequency of each character in the training set.
    :param max_word_length: Maximum length of words in the training set.
    :param device: Device on which the model is running.
    :param guessed_chars: Set of characters already guessed.
    :param fallback_strategy: Whether to use fallback strategy or not.
    :return: Guessed character.
    """

    # Preprocess the masked word
    feature_set, missed_chars = process_single_word_inference(masked_word, char_frequency, max_word_length, normalize=True)
    feature_set, missed_chars = feature_set.unsqueeze(0).to(device), missed_chars.unsqueeze(0).to(device)
    sequence_lengths = torch.tensor([feature_set.size(1)], dtype=torch.long) # .to(device)

    # Get model output
    with torch.no_grad():
        output = model(feature_set, sequence_lengths, missed_chars)
    
    # Get probabilities of the last character position
    last_char_position = sequence_lengths.item() - 1
    probabilities = torch.softmax(output[0, last_char_position, :], dim=-1)

    # Exclude already guessed characters and get the most probable character
    probabilities[torch.tensor([char_to_idx[char] for char in guessed_chars], \
        dtype=torch.long)] = 0
    best_char_index = torch.argmax(probabilities).item()
    guessed_char = idx_to_char[best_char_index]
    # guessed_char = get_random_character() #TODO

    # Fallback strategy: choose the most common unguessed character
    if fallback_strategy and guessed_char in \
        guessed_chars or guessed_char == '_':
        sorted_chars = sorted(char_frequency.items(), \
            key=lambda x: x[1], reverse=True)
        for char, _ in sorted_chars:
            if char not in guessed_chars:
                guessed_char = char
                break

    return guessed_char


# # Example usage
# guessed_char = get_random_character()
# print(guessed_char)

# # def gues(self, word):
def guess(model, word, char_frequency, \
    max_word_length, device, guessed_letters):
    
    cleaned_word = "".join(char.lower() \
        for char in word if char.isalpha() or char == '_')

    # Predict the next character using the updated guess_character function
    guessed_char =  guess_character(
        model, cleaned_word,
        char_frequency,
        max_word_length,
        device,
        guessed_letters  # Pass the list of guessed letters
    )

    # guessed_char =  get_random_character()

    # Add the new guess to the guessed letters list
    if guessed_char not in guessed_letters:
        guessed_letters.append(guessed_char)
    
    return guessed_char

# mimic api
def update_word_state(actual_word, current_state, guessed_char):
    return ''.join([guessed_char if actual_word[i] == guessed_char else \
        current_state[i] for i in range(len(actual_word))])

def play_game_with_a_word(model, word, char_frequency, \
    max_word_length, device, max_attempts=6, normalize=True):
    guessed_letters = []  # A list to keep track of guessed characters
    attempts_remaining = max_attempts
    masked_word = "_" * len(word)
    game_status = "ongoing"

    # print(f"Starting the game. Word to guess: {' '.join(masked_word)}")  # Display initial state

    while game_status == "ongoing" and attempts_remaining > 0:
        guessed_char = guess(model, word, char_frequency, max_word_length, device, guessed_letters)
        guessed_letters.append(guessed_char)  # Add to the list of guessed letters
        masked_word = update_word_state(word, masked_word, guessed_char)

        if guessed_char not in word:
            attempts_remaining -= 1

        # print(f"Guessed: {guessed_char}, Word: {' '.join(masked_word)}, Attempts remaining: {attempts_remaining}")  # Display current state

        if "_" not in masked_word:
            game_status = "success"

    # if game_status == "success":
    #     # print(f"Congratulations! The word '{word}' was guessed correctly.")
    # else:
    #     # print(f"Game over. The correct word was '{word}'.")

    return masked_word == word, \
        masked_word, max_attempts - attempts_remaining

#### Dont change above this ######

import random

def generate_masked_word_variants(word, max_variants, max_masks):
    word_length = len(word)
    indices = list(range(word_length))
    masked_versions = set()

    while len(masked_versions) < max_variants:
        num_masks = random.randint(1, max_masks)  # Randomly choose the number of masks
        mask_indices = set(random.sample(indices, num_masks))
        masked_word = ''.join(c if i not in mask_indices \
            else '_' for i, c in enumerate(word))
        masked_versions.add(masked_word)

    return list(masked_versions)

def process_word(word):
    word_length = len(word)
    max_variants = min(word_length, 10)  # Limit the number of variants

    # Ensure at least one character is always revealed, except for the fully masked state
    max_masks = max(1, int(word_length * 0.5))

    # Generate other masked variants
    other_masked_states = generate_masked_word_variants(word, max_variants - 1, max_masks)

    # Ensure a fully masked state is always included
    initial_states = ['_' * word_length] + other_masked_states

    return initial_states


import random

def simulate_game_progress(model, word, \
    initial_state, char_frequency, max_word_length, \
        device, max_attempts=6, normalize=True, difficulty="medium", \
            outcome_preference=None):
    
    guessed_letters = []
    attempts_remaining = max_attempts
    masked_word = initial_state  # Starting with the initial masked state of the word

    # Set difficulty level
    difficulty_levels = {"easy": 0.8, "medium": 0.5, "hard": 0.2}
    correct_guess_chance = difficulty_levels.get(difficulty, 0.5)

    game_progress = []

    while attempts_remaining > 0:
        possible_correct_guesses = [char for char in word if char not in guessed_letters and char.isalpha()]
        possible_incorrect_guesses = [char for char in 'abcdefghijklmnopqrstuvwxyz' if char not in word and char not in guessed_letters]

        # Check if no more characters left to guess
        if not possible_correct_guesses and not possible_incorrect_guesses:
            print("No more characters left to guess.")
            break

        if outcome_preference == "win" and possible_correct_guesses:
            guessed_char = random.choice(possible_correct_guesses) \
                if random.random() < correct_guess_chance else random.choice(possible_incorrect_guesses)
        elif outcome_preference == "lose" and possible_incorrect_guesses:
            guessed_char = random.choice(possible_incorrect_guesses) \
                if random.random() > correct_guess_chance else random.choice(possible_correct_guesses)
        else:
            guessed_char = random.choice(possible_correct_guesses \
                + possible_incorrect_guesses) if \
                    (possible_correct_guesses + possible_incorrect_guesses) else None

        if not guessed_char:
            continue

        guessed_letters.append(guessed_char)
        new_masked_word = update_word_state(word, masked_word, guessed_char)
        correct_guess = guessed_char in word

        game_progress.append((guessed_char, new_masked_word, correct_guess))

        if new_masked_word == word:
            return True, game_progress
        elif not correct_guess:
            attempts_remaining -= 1

        masked_word = new_masked_word

    return masked_word == word, game_progress





















# def simulate_game_progress(model, word, initial_state, \
#     char_frequency, max_word_length, device, max_attempts=6, \
#         normalize=True, difficulty="medium", outcome_preference=None):

#     guessed_letters = [char for char in initial_state if char != '_']  # Include letters from the initial state
#     # print(guessed_letters)
#     attempts_remaining = max_attempts
#     masked_word = initial_state  # Starting with the initial masked state of the word

#     # Set difficulty level
#     difficulty_levels = {"easy": 0.8, "medium": 0.5, "hard": 0.2}
#     correct_guess_chance = difficulty_levels.get(difficulty, 0.5)

#     game_progress = []

#     while attempts_remaining > 0:
#         # Choose the guessed character based on the outcome preference and difficulty
#         if outcome_preference == "win":
#             guessed_char = random.choice(word) if random.random() \
#                 < correct_guess_chance else random.choice([char for \
#                     char in 'abcdefghijklmnopqrstuvwxyz' if char not in guessed_letters])
#         elif outcome_preference == "lose":
#             guessed_char = random.choice([char for \
#                 char in 'abcdefghijklmnopqrstuvwxyz' if char not in word and \
#                     char not in guessed_letters]) if random.random() \
#                         > correct_guess_chance else random.choice([char for \
#                             char in word if char not in guessed_letters])
#         else:
#             # Normal guessing behavior without a specific outcome preference
#             guessed_char = random.choice([char for \
#                 char in 'abcdefghijklmnopqrstuvwxyz' if char not in guessed_letters])

#         # Avoid repeating guesses
#         if guessed_char in guessed_letters:
#             continue

#         guessed_letters.append(guessed_char)
#         new_masked_word = update_word_state(word, masked_word, guessed_char)
#         correct_guess = guessed_char in word

#         game_progress.append((guessed_char, new_masked_word, correct_guess))

#         if new_masked_word == word:  # Word guessed correctly
#             return True, game_progress
#         elif not correct_guess:  # Incorrect guess
#             attempts_remaining -= 1

#         if attempts_remaining == 0:
#             return False, game_progress

#         masked_word = new_masked_word  # Update for next iteration

#     return masked_word == word, game_progress






# def simulate_game_progress(model, word, char_frequency, \
#     max_word_length, device, max_attempts=6, \
#         normalize=True, \
#         outcome_preference=None):
#     guessed_letters = []
#     attempts_remaining = max_attempts
#     masked_word = "_" * len(word)
#     game_progress = []
#     game_outcome = random.choice(["win", "lose"]) if \
#         outcome_preference is None else outcome_preference

#     while attempts_remaining > 0:
#         guessed_char = guess(model, word, char_frequency, max_word_length, \
#             device, guessed_letters)
#         guessed_letters.append(guessed_char)
#         new_masked_word = update_word_state(word, masked_word, guessed_char)
#         correct_guess = guessed_char in word

#         game_progress.append((guessed_char, new_masked_word, correct_guess))

#         if game_outcome == "win" and new_masked_word == word:
#             return True, game_progress
#         elif game_outcome == "lose" and not correct_guess:
#             attempts_remaining -= 1
#             if attempts_remaining == 0:
#                 return False, game_progress

#         masked_word = new_masked_word

#     return masked_word == word, game_progress

# def simulate_game_progress(model, word, \
#     char_frequency, max_word_length, device, \
#         max_attempts=6, normalize=True):
#     guessed_letters = []
#     attempts_remaining = max_attempts
#     masked_word = "_" * len(word)
#     game_progress = []  # This will store the detailed progression of the game

#     while attempts_remaining > 0:
#         guessed_char = guess(model, word, char_frequency, max_word_length, device, guessed_letters)
#         guessed_letters.append(guessed_char)
#         new_masked_word = update_word_state(word, masked_word, guessed_char)
#         correct_guess = guessed_char in word

#         # Record each step of the game
#         game_progress.append((guessed_char, new_masked_word, correct_guess))

#         if new_masked_word == word:
#             return True, game_progress  # Return the entire game progress
#         elif not correct_guess:
#             attempts_remaining -= 1

#         masked_word = new_masked_word

#     return False, game_progress  # Return the entire game progress


