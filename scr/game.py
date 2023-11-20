import random
import re
import string
from collections import Counter

import torch
import torch.nn as nn

# process_single_word, get_missed_characters, char_to_idx, idx_to_char
from scr.feature_engineering import *
from scr.feature_engineering import (
    char_to_idx,
    idx_to_char,
    process_single_word_inference,
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_character():
    # Choose from lowercase letters
    characters = 'a'  # string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
    random_char = random.choice(characters)
    return random_char


# # def gues(self, word):
def guess_character(model, masked_word, char_frequency,
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
    feature_set, missed_chars = process_single_word_inference(
        masked_word, char_frequency, max_word_length, normalize=True)
    feature_set, missed_chars = feature_set.unsqueeze(
        0).to(device), missed_chars.unsqueeze(0).to(device)
    sequence_lengths = torch.tensor(
        [feature_set.size(1)], dtype=torch.long)  # .to(device)

    # Get model output
    with torch.no_grad():
        output = model(feature_set, sequence_lengths, missed_chars)

    # Get probabilities of the last character position
    last_char_position = sequence_lengths.item() - 1
    probabilities = torch.softmax(output[0, last_char_position, :], dim=-1)

    # Exclude already guessed characters and get the most probable character
    probabilities[torch.tensor([char_to_idx[char] for char in guessed_chars],
                               dtype=torch.long)] = 0
    best_char_index = torch.argmax(probabilities).item()
    guessed_char = idx_to_char[best_char_index]
    # guessed_char = get_random_character() #TODO

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


# # Example usage
# guessed_char = get_random_character()
# print(guessed_char)

# # def gues(self, word):
def guess(model, word, char_frequency,
          max_word_length, device, guessed_letters):

    cleaned_word = "".join(char.lower()
                           for char in word if char.isalpha() or char == '_')

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
                          max_word_length, device, max_attempts=6, normalize=True):
    guessed_letters = []  # A list to keep track of guessed characters
    attempts_remaining = max_attempts
    masked_word = "_" * len(word)
    game_status = "ongoing"

    # print(f"Starting the game. Word to guess: {' '.join(masked_word)}")  # Display initial state

    while game_status == "ongoing" and attempts_remaining > 0:
        guessed_char = guess(model, word, char_frequency,
                             max_word_length, device, guessed_letters)
        # Add to the list of guessed letters
        guessed_letters.append(guessed_char)
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
    unique_chars = list(set(word))  # Convert set to list
    masked_versions = set()

    count = 0
    while count < max_variants:
        num_chars_to_reveal = random.randint(1, min(max_masks, len(unique_chars)))
        chars_to_reveal = random.sample(unique_chars, num_chars_to_reveal)
        masked_word = ''.join(c if c in chars_to_reveal else '_' for c in word)

        if masked_word not in masked_versions:
            yield masked_word  # Yield each unique masked word variant
            masked_versions.add(masked_word)
            count += 1

def process_word(word, mask_prob=0.9, max_variants=10):
    word_length = len(word)
    max_variants = min(word_length, max_variants)
    max_masks = max(1, int(len(set(word)) * mask_prob))

    return ['_' * word_length] + list(generate_masked_word_variants(word, \
        max_variants - 1, max_masks))



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
                                       if masked_word[idx] == '_' and char not \
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