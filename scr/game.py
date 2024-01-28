import random
import re
import string
from collections import Counter
from itertools import combinations

import torch
import torch.nn as nn
from IPython.display import clear_output, display
# from tqdm import tqdm
from tqdm.notebook import tqdm

# process_single_word, get_missed_characters, char_to_idx, idx_to_char
from scr.feature_engineering import *
from scr.guess import *
from scr.utils import *

# from scr.feature_engineering import (char_to_idx, idx_to_char,
#                                      process_single_word_inference)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def determine_current_state(masked_word, guessed_chars):
#     word_length = len(masked_word)
#     num_revealed_chars = sum(1 for char in masked_word if char != '_')
#     num_guessed_unique_chars = len(set(guessed_chars))

#     # Estimate the state based on the number of revealed characters
#     if num_revealed_chars == 0:
#         return "allMasked"
#     elif num_revealed_chars <= 1 and word_length > 4:
#         return "early"
#     elif num_revealed_chars <= num_guessed_unique_chars // 4:
#         return "quarterRevealed"
#     elif num_revealed_chars <= num_guessed_unique_chars // 2:
#         return "midRevealed"
#     elif num_revealed_chars <= 3 * num_guessed_unique_chars // 4:
#         return "midLateRevealed"
#     elif num_revealed_chars < num_guessed_unique_chars:
#         return "lateRevealed"
#     else:
#         return "nearEnd"


# mimic api


def update_word_state(actual_word, current_state, guessed_char):
    return ''.join([guessed_char if actual_word[i] == guessed_char else
                    current_state[i] for i in range(len(actual_word))])


def play_game_with_a_word(model, word, char_frequency,
                          max_word_length, max_attempts=6,
                          normalize=True):

    guessed_letters = []  # A list to keep track of guessed characters
    attempts_remaining = max_attempts
    masked_word = "_" * len(word)
    # print(masked_word)
    game_status = "ongoing"

    # print(f"Starting the game. Word to guess: {' '.join(masked_word)}")  # Display initial state

    while game_status == "ongoing" and attempts_remaining > 0:

        guessed_char = guess(model, masked_word, char_frequency,
                             max_word_length, guessed_letters)

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


def play_games_and_calculate_stats(model, words_list, char_frequency,
                                   max_word_length, max_attempts=6):
    stats = {}
    total_wins = 0
    total_games = 0
    total_attempts_used = 0

    # For length-wise details
    length_wise_stats = {}

    # # Initialize progress bar
    # with tqdm(total=len(words_list), desc="Processing words", unit="word") as pbar:
    # Initialize progress bar
    # with tqdm(total=len(words_list)) as pbar:
    with tqdm(total=len(words_list), desc="Processing words", unit="word", leave=False) as pbar:
        for word in words_list:
            word = word.lower()  # Ensure the word is in lowercase
            word_length = len(word)

            # Initialize length-wise stats
            if word_length not in length_wise_stats:
                length_wise_stats[word_length] = {
                    "total_games": 0, "wins": 0, "total_attempts_used": 0}

            # Initialize stats for this word
            if word not in stats:
                stats[word] = {"total_games": 0,
                               "wins": 0, "total_attempts_used": 0}

                # Play game with the word
                win, masked_word, attempts_used = play_game_with_a_word(
                    model, word, char_frequency, max_word_length, max_attempts
                )

                # Update stats for this word and length-wise stats
                stats[word]["total_games"] += 1
                stats[word]["total_attempts_used"] += attempts_used
                length_wise_stats[word_length]["total_games"] += 1
                length_wise_stats[word_length]["total_attempts_used"] += attempts_used

                if win:
                    stats[word]["wins"] += 1
                    length_wise_stats[word_length]["wins"] += 1
                    total_wins += 1

                # # Update progress bar
                # pbar.update(1)

            total_games += 1
            total_attempts_used += attempts_used

            # # # Update progress bar
            pbar.update(1)

    # pbar.close()
    # clear_output(wait=True)

    # Calculate win rate and average attempts used for each word
    for word, data in stats.items():
        data["win_rate"] = (data["wins"] / data["total_games"]) * 100
        data["average_attempts_used"] = data["total_attempts_used"] / \
            data["total_games"]

    # Calculate length-wise stats
    for length, data in length_wise_stats.items():
        data["win_rate"] = (data["wins"] / data["total_games"]) * 100
        data["average_attempts_used"] = data["total_attempts_used"] / \
            data["total_games"]

    overall_win_rate = (total_wins / total_games) * 100
    overall_avg_attempts = total_attempts_used / total_games

    return {
        "stats": stats,
        "overall_win_rate": overall_win_rate,
        "overall_avg_attempts": overall_avg_attempts,
        "length_wise_stats": length_wise_stats
    }


#### Dont change above this ######


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
