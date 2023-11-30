import random
import re
import string
from collections import Counter
from itertools import combinations

import torch
import torch.nn as nn

# process_single_word, get_missed_characters, char_to_idx, idx_to_char
from scr.feature_engineering import *
from scr.feature_engineering import (char_to_idx, idx_to_char,
                                     process_single_word_inference)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_character():
    # Choose from lowercase letters
    characters = 'a'  # string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
    random_char = random.choice(characters)
    return random_char


# # # def gues(self, word):
def guess_character(model, masked_word, char_frequency,
                    max_word_length, device, guessed_chars,
                    max_seq_length=1, fallback_strategy=True):
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
    # print(f'masked word: ', masked_word)

    batch_features, batch_missed_characters \
        = process_batch_of_games([masked_word],
                                 char_frequency, max_word_length, max_seq_length)

    batch_size = 1

    sequence_lengths = torch.tensor([max_seq_length]
                                    * batch_size, dtype=torch.long).cpu()

    # batch_features = batch_features.to(device)
    # # sequence_lengths = sequence_lengths.to(device)
    # batch_missed_characters = batch_missed_characters.to(device)

    with torch.no_grad():
        output = model(batch_features, sequence_lengths,
                       batch_missed_characters)  # .to(device)

    # Assuming the last character in the sequence is the current guess
    last_char_position = sequence_lengths.item() - 1
    probabilities = torch.softmax(output[0, last_char_position, :], dim=-1)

    # Exclude already guessed characters
    guessed_indices = [char_to_idx[char]
                       for char in guessed_chars if char in char_to_idx]
    # print(device)
    probabilities[torch.tensor(
        guessed_indices, dtype=torch.long, device=device)] = 0

    # Find the best character to guess
    best_char_index = torch.argmax(probabilities).item()
    guessed_char = idx_to_char[best_char_index]

    # guessed_char = get_random_character() # TODO

    # Fallback strategy: choose the most common unguessed character
    if fallback_strategy and guessed_char in \
            guessed_chars or guessed_char == '_':
        sorted_chars = sorted(char_frequency.items(),
                              key=lambda x: x[1], reverse=True)
        for char, _ in sorted_chars:
            if char not in guessed_chars:
                guessed_char = char
                break

    return guessed_char

# # def gues(self, word):


def guess(model, word, char_frequency,
          max_word_length, device, guessed_letters):

    cleaned_word = "".join(char.lower()
                           for char in word if char.isalpha() or char == '_')
    # print(cleaned_word)

    # Predict the next character using the updated guess_character function
    guessed_char = guess_character(
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
    return ''.join([guessed_char if actual_word[i] == guessed_char else
                    current_state[i] for i in range(len(actual_word))])


def play_game_with_a_word(model, word, char_frequency,
                          max_word_length, device, max_attempts=6,
                          normalize=True):
    guessed_letters = []  # A list to keep track of guessed characters
    attempts_remaining = max_attempts
    masked_word = "_" * len(word)
    # print(masked_word)
    game_status = "ongoing"

    # print(f"Starting the game. Word to guess: {' '.join(masked_word)}")  # Display initial state

    while game_status == "ongoing" and attempts_remaining > 0:
        guessed_char = guess(model, masked_word, char_frequency,
                             max_word_length, device, guessed_letters)
        # Add to the list of guessed letters
        guessed_letters.append(guessed_char)
        masked_word = update_word_state(word, masked_word, guessed_char)

        if guessed_char not in word:
            attempts_remaining -= 1

        # print(f"Guessed: {guessed_char}, Word: {' '.join(masked_word)}, \
        # Attempts remaining: {attempts_remaining}")  # Display current state

        if "_" not in masked_word:
            game_status = "success"

    # if game_status == "success":
    #     # print(f"Congratulations! The word '{word}' was guessed correctly.")
    # else:
    #     # print(f"Game over. The correct word was '{word}'.")

    return masked_word == word, \
        masked_word, max_attempts - attempts_remaining

#### Dont change above this ######

# import random

# def generate_masked_word_variants(word, max_variants, max_masks):
#     word_length = len(word)
#     unique_chars = list(set(word))  # Convert set to list
#     masked_versions = set()

#     count = 0
#     while count < max_variants:
#         num_chars_to_reveal = random.randint(1, min(max_masks, \
#             len(unique_chars)))
#         chars_to_reveal = random.sample(unique_chars, num_chars_to_reveal)
#         masked_word = ''.join(c if c in chars_to_reveal else '_' \
#             for c in word)

#         if masked_word not in masked_versions:
#             yield masked_word  # Yield each unique masked word variant
#             masked_versions.add(masked_word)
#             count += 1

# def process_word(word, mask_prob=0.9, max_variants=10):
#     word_length = len(word)
#     max_variants = min(word_length, max_variants)
#     max_masks = max(1, int(len(set(word)) * mask_prob))

#     return ['_' * word_length] + list(generate_masked_word_variants(word, \
#         max_variants - 1, max_masks))


def optimized_masked_variants(word, max_variants, max_masks):
    unique_chars = list(set(word))
    all_combinations = []

    # Generate all possible combinations of characters to reveal
    for r in range(1, min(max_masks, len(unique_chars)) + 1):
        all_combinations.extend(combinations(unique_chars, r))

    # Randomly select combinations and generate masked words
    random.shuffle(all_combinations)
    masked_versions = set()
    for chars_to_reveal in all_combinations[:max_variants]:
        masked_word = ''.join(c if c in chars_to_reveal
                              else '_' for c in word)
        if masked_word not in masked_versions:
            masked_versions.add(masked_word)
            yield masked_word
        if len(masked_versions) >= max_variants:
            break


def process_word(word, mask_prob=0.9, max_variants=10):
    word_length = len(word)
    max_masks = max(1, int(len(set(word)) * mask_prob))
    return ['_' * word_length] + list(optimized_masked_variants(word,
                                                                max_variants - 1, max_masks))


def process_word_for_six_states(word):
    word_length = len(word)
    unique_chars = list(set(word))  # List of unique characters in the word
    random.shuffle(unique_chars)  # Shuffle to get a random order

    # Calculate the number of characters to reveal for each state
    num_chars_revealed = {
        "allMasked": 0,
        "early": 1 if word_length > 4 else 0,
        "quarterRevealed": len(unique_chars) // 4,
        "midRevealed": len(unique_chars) // 2,
        "midLateRevealed": 3 * len(unique_chars) // 4,
        "lateRevealed": max(1, len(unique_chars) - 1),
        "nearEnd": len(unique_chars) - 1
    }

    masked_states = {}
    chars_revealed_so_far = set()

    for state, num_to_reveal in num_chars_revealed.items():
        while len(chars_revealed_so_far) < num_to_reveal:
            chars_revealed_so_far.add(unique_chars[len(chars_revealed_so_far)])
        masked_word = ''.join(
            char if char in chars_revealed_so_far else '_' for char in word)
        masked_states[state] = masked_word

    return masked_states


def simulate_game_progress(model, word, initial_state, char_frequency,
                           max_word_length, device,
                           max_attempts=6, normalize=True,
                           difficulty="medium", outcome_preference=None):

    guessed_letters = set()
    attempts_remaining = max_attempts
    masked_word = initial_state
    correct_guess_chance = {"easy": 0.8,
                            "medium": 0.5, "hard": 0.2}[difficulty]
    game_progress = []

    # print(
    #     f"Starting game. Word: {word}, Initial State: {initial_state}, Attempts: {max_attempts}")

    while attempts_remaining > 0 and '_' in masked_word:
        guessed_letters.update(char for idx, char in enumerate(
            word) if masked_word[idx] != '_')

        possible_correct_guesses = set(char for idx, char in enumerate(word)
                                       if masked_word[idx] == '_' and char not
                                       in guessed_letters)
        possible_incorrect_guesses = set(
            'abcdefghijklmnopqrstuvwxyz') - set(word) - guessed_letters

        # print(
        #     f"Current state: {masked_word}, Attempts remaining: {attempts_remaining}")
        # print(f"Guessed letters so far: {guessed_letters}")

        guessed_char = None
        if outcome_preference == "win" and possible_correct_guesses:
            guessed_char = (random.choice(list(possible_correct_guesses))
                            if random.random() < correct_guess_chance
                            else random.choice(list(possible_incorrect_guesses))
                            if possible_incorrect_guesses else None)
        elif outcome_preference == "lose" and possible_incorrect_guesses:
            guessed_char = (random.choice(list(possible_incorrect_guesses))
                            if random.random() > correct_guess_chance
                            else random.choice(list(possible_correct_guesses))
                            if possible_correct_guesses else None)
        else:
            all_choices = list(possible_correct_guesses) + \
                list(possible_incorrect_guesses)
            guessed_char = random.choice(all_choices) if all_choices else None

        if not guessed_char:
            # print("No valid character to guess, skipping turn.")
            continue

        # Check if the character has been guessed before
        if guessed_char in guessed_letters:
            # print(
            #     f"Character '{guessed_char}' has been guessed before, skipping.")
            continue

        correct_guess = guessed_char in word
        guessed_letters.add(guessed_char)
        # print(f"Guessed '{guessed_char}', Correct: {correct_guess}")

        if correct_guess:
            masked_word = update_word_state(word, masked_word, guessed_char)
            game_progress.append((guessed_char, masked_word, True))
            # print(f"Correct guess. New state: {masked_word}")
        else:
            attempts_remaining -= 1
            game_progress.append((guessed_char, masked_word, False))
    #         print(
    #             f"Incorrect guess. New state: {masked_word}, Attempts remaining: {attempts_remaining}")

    # print(
    #     f"Game ended. Word {'guessed correctly' if masked_word == word else 'not guessed correctly'}. Final State: {masked_word}")

    return masked_word == word, game_progress


def update_word_state(word, masked_word, guessed_char):
    return "".join(char if char == guessed_char or masked_word[idx] != '_'
                   else masked_word[idx] for idx, char in enumerate(word))
