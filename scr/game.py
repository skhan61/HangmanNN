import random
import re
import string
from collections import Counter
from itertools import combinations

import torch
import torch.nn as nn
from IPython.display import clear_output, display
from tqdm.auto import tqdm
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

def update_word_state(word, masked_word, guessed_char):
    return "".join(char if char == guessed_char or masked_word[idx] != '_'
                   else masked_word[idx] for idx, char in enumerate(word))


# def update_word_state(actual_word, current_state, guessed_char):
#     return ''.join([guessed_char if actual_word[i] == guessed_char else
#                     current_state[i] for i in range(len(actual_word))])

def play_game_with_a_word(model, word, char_frequency,
                          max_word_length, max_attempts=6, normalize=True):
    guessed_letters = []  # Track guessed characters
    attempts_remaining = max_attempts
    masked_word = "_" * len(word)
    game_status = "ongoing"
    correct_guesses = 0
    incorrect_guesses = 0
    seq_len_stats = []  # Enhanced to track more detailed game-level statistics

    while game_status == "ongoing" and attempts_remaining > 0:
        guessed_char = guess(model, masked_word, char_frequency,
                             max_word_length, guessed_letters)
        if guessed_char not in guessed_letters:
            guessed_letters.append(guessed_char)

        previous_masked_word = masked_word
        masked_word = update_word_state(word, masked_word, guessed_char)

        correct_guess = guessed_char in word
        if correct_guess:
            correct_guesses += 1
        else:
            incorrect_guesses += 1
            attempts_remaining -= 1

        seq_len_stats.append({
            "masked_word": masked_word,
            "guess": guessed_char,
            "correct_guess": correct_guess,
            "attempts_remaining": attempts_remaining,
            "uncovered_letters": sum(1 for orig, new in zip(word, masked_word) if orig == new and new != '_'),
            "correct_guesses_so_far": correct_guesses,
            "incorrect_guesses_so_far": incorrect_guesses,
            # Copy to avoid reference issues
            "guessed_letters": list(guessed_letters)
        })

        if "_" not in masked_word:
            game_status = "success"

    # Existing detailed sequence length evolution summary
    seq_len_evolution_summary = {
        "total_guesses": len(seq_len_stats),
        "seq_length_evolution": [stat['uncovered_letters'] for stat in seq_len_stats],
        "final_uncovered_letters": seq_len_stats[-1]['uncovered_letters'] if seq_len_stats else 0,
        "correct_guesses": correct_guesses,
        "incorrect_guesses": incorrect_guesses
    }

    # New simplified summary
    simplified_summary = {
        "total_guesses": len(seq_len_stats),
        "final_uncovered_letters": seq_len_stats[-1]['uncovered_letters'] if seq_len_stats else 0,
        "game_won": game_status == "success"
    }

    # Return the existing details along with the new simplified summary
    return masked_word == word, masked_word, max_attempts - attempts_remaining, \
        seq_len_stats, seq_len_evolution_summary, simplified_summary


def play_games_and_calculate_stats(model, words_list, char_frequency,
                                   max_word_length, max_attempts=6):
    stats = {}
    total_wins = 0
    total_games = 0
    total_attempts_used = 0
    length_wise_stats = {}
    seq_len_level_stats = {}  # Aggregate stats at sequence length level
    # Collect summaries of sequence length evolution for each game
    game_evolution_summaries = []
    simplified_summaries = []  # Collect simplified summaries for each game

    with tqdm(total=len(words_list), desc="Processing words", unit="word", leave=False) as pbar:
        for word in words_list:
            word = word.lower()
            word_length = len(word)
            if word_length not in length_wise_stats:
                length_wise_stats[word_length] = {
                    "total_games": 0, "wins": 0, "total_attempts_used": 0}
            if word not in stats:
                stats[word] = {"total_games": 0,
                               "wins": 0, "total_attempts_used": 0}

                # Adjusted to include simplified_summary in the unpacked values
                win, masked_word, attempts_used, game_seq_len_stats, seq_len_evolution_summary, simplified_summary \
                    = play_game_with_a_word(model, word, char_frequency, max_word_length, max_attempts)

                game_evolution_summaries.append(seq_len_evolution_summary)
                # Store each game's simplified summary
                simplified_summaries.append(simplified_summary)

                for step_stat in game_seq_len_stats:
                    seq_len = step_stat['uncovered_letters']
                    if seq_len not in seq_len_level_stats:
                        seq_len_level_stats[seq_len] = {
                            "total_attempts": 0, "correct_guesses": 0, "games": 0}
                    seq_len_level_stats[seq_len]["total_attempts"] += 1
                    # Count games that reached this sequence length
                    seq_len_level_stats[seq_len]["games"] += 1
                    if step_stat['correct_guess']:
                        seq_len_level_stats[seq_len]["correct_guesses"] += 1

                stats[word]["total_games"] += 1
                stats[word]["total_attempts_used"] += attempts_used
                length_wise_stats[word_length]["total_games"] += 1
                length_wise_stats[word_length]["total_attempts_used"] += attempts_used

                if win:
                    stats[word]["wins"] += 1
                    length_wise_stats[word_length]["wins"] += 1
                    total_wins += 1

            total_games += 1
            total_attempts_used += attempts_used
            pbar.update(1)

    # Calculate win rates and average attempts for sequence lengths
    for seq_len, data in seq_len_level_stats.items():
        data["win_rate"] = (data["correct_guesses"] / data["total_attempts"]
                            ) * 100 if data["total_attempts"] > 0 else 0
        data["average_attempts"] = data["total_attempts"] / \
            data["games"] if data["games"] > 0 else 0

    for word, data in stats.items():
        data["win_rate"] = (data["wins"] / data["total_games"]) * 100
        data["average_attempts_used"] = data["total_attempts_used"] / \
            data["total_games"]

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
        "length_wise_stats": length_wise_stats,
        "seq_len_level_stats": seq_len_level_stats,
        "game_evolution_summaries": game_evolution_summaries,
        # Include simplified summaries in the return value
        "simplified_summaries": simplified_summaries
    }


def calculate_mean_win_rate_by_seq_length(simplified_summaries):
    # Dictionary to hold the total number of wins and games for each unique total_guesses value
    guess_stats = {}

    # Iterate through each game summary
    for summary in simplified_summaries:
        total_guesses = summary['total_guesses']
        game_won = summary['game_won']

        # Initialize the dictionary entry if it doesn't exist
        if total_guesses not in guess_stats:
            guess_stats[total_guesses] = {'wins': 0, 'games': 0}

        # Increment the win count if the game was won
        if game_won:
            guess_stats[total_guesses]['wins'] += 1

        # Increment the total game count
        guess_stats[total_guesses]['games'] += 1

    # Calculate the mean win rate for each unique total_guesses value
    mean_win_rates = {}
    for guesses, stats in guess_stats.items():
        win_rate = (stats['wins'] / stats['games']) * \
            100  # Win rate as a percentage
        mean_win_rates[guesses] = win_rate

    return mean_win_rates


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


# def update_word_state(word, masked_word, guessed_char):
#     return "".join(char if char == guessed_char or masked_word[idx] != '_'
#                    else masked_word[idx] for idx, char in enumerate(word))
