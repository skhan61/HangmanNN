import gc
import random
from collections import Counter

import numpy as np
import torch  # Ensure torch is imported for tensor operations
import torch.nn.functional as F

# 1. Common Utility Functions
# Character set and mapping
# Add blank character '' to the character set
char_to_idx = {char: idx for idx, char in enumerate(
    ['', '_'] + list('abcdefghijklmnopqrstuvwxyz'))}

idx_to_char = {idx: char for char, idx in char_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game_states = ['allMasked', 'early', 'quarterRevealed', 'midRevealed',
               'midLateRevealed', 'lateRevealed', 'nearEnd']
game_state_to_idx = {state: idx for idx, state in enumerate(game_states)}


# Function to convert batch tensor to missed characters
def batch_to_chars(batch):
    from scr.feature_engineering import idx_to_char
    missed_chars = []
    for sample in batch:
        chars = ''
        for idx, char_present in enumerate(sample[0]):
            if char_present == 0:
                chars += idx_to_char[idx]
        missed_chars.append(chars)
    return missed_chars


def encode_word(word):
    # print(word)
    return [char_to_idx[char] for char in word]


def get_missed_characters(word):
    # print(f'print word form get missed char: ', word)
    # Check if the input is a list and extract the first element if so
    if isinstance(word, list) and len(word) > 0:
        word = word[0]

    # print(word)
    all_chars = set(char_to_idx.keys())
    present_chars = set(word)
    missed_chars = all_chars - present_chars
    return torch.tensor([1 if char in missed_chars else
                         0 for char in char_to_idx],
                        dtype=torch.float)


def calculate_char_frequencies(word_list):
    char_counts = Counter(''.join(word_list))
    return {char: char_counts[char] / sum(char_counts.values())
            for char in char_to_idx}


def pad_tensor(tensor, length):
    return torch.cat([tensor, torch.zeros(length - len(tensor))], dim=0)


def encode_ngrams(ngrams, n):
    # print(f"Encoding n-grams: {ngrams}")
    encoded_ngrams = [char_to_idx[char] for ngram in ngrams for char in ngram]
    fixed_length = n * 2
    encoded = encoded_ngrams[:fixed_length] + \
        [0] * (fixed_length - len(encoded_ngrams))
    # print(f"Encoded n-grams (fixed length): {encoded}")
    return encoded


def calculate_word_frequencies(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}


def extract_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]


# 3. Inference Data Preparation
# Dont not change

def encode_guessed_letters(guessed_letters, char_to_idx):
    # Initialize the encoded vector with zeros
    # The size is based on the length of the char_to_idx dictionary
    encoded = [0] * len(char_to_idx)

    for letter in guessed_letters:
        # Handle special cases: empty and underscore
        if letter == '':
            index = char_to_idx['']
        elif letter == '_':
            index = char_to_idx['_']
        else:
            # Standard alphabet
            index = char_to_idx.get(letter.lower(), 0)

        # Set the corresponding index to 1
        encoded[index] = 1

    return encoded
# =============================================================


def analyze_and_extract_features(game_states,
                                 guesses,
                                 maximum_word_length=None):
    total_attempts = 0
    total_correct_guesses = 0
    guess_outcomes = []  # Tracks the success (1) or failure (0) of each guess
    cumulative_guessed_letters = []  # To track all guessed letters cumulatively
    longest_success_streak = 0
    current_streak = 0
    missed_guesses = 0
    duplicate_guesses = 0
    guess_diversity = 0
    critical_letters = 'aeiou'
    critical_letter_uncover_count = 0

    for i, guess in enumerate(guesses):
        remaining_attempts = 6 - total_attempts
        # print(
        #     f"\nAnalyzing guess {i+1}: '{guess}', Remaining attempts: {remaining_attempts}")

        if remaining_attempts <= 0:
            # print("No attempts remaining. Stopping analysis.")
            break

        cumulative_guessed_letters.append(guess)

        
        prev_state = game_states[i]
        current_state = game_states[i + 1] if i + \
            1 < len(game_states) else game_states[-1]

        if maximum_word_length is None:
            maximum_word_length = len(current_state)
        comparison_length = min(len(current_state), maximum_word_length)
        new_letters_this_guess = 0

        for j in range(comparison_length):
            if prev_state[j] == '_' and current_state[j] != '_':
                new_letters_this_guess += 1
                if current_state[j] in critical_letters:
                    critical_letter_uncover_count += 1

        if new_letters_this_guess > 0:
            total_correct_guesses += 1
            guess_outcomes.append(1)
            current_streak += 1
            # print(
            #     f"Correct guess! New letters revealed: {new_letters_this_guess}.")
        else:
            total_attempts += 1
            missed_guesses += 1
            guess_outcomes.append(0)
            current_streak = 0  # Reset streak
            # print("Incorrect guess. No new letters revealed.")

        longest_success_streak = max(longest_success_streak, current_streak)

        if cumulative_guessed_letters.count(guess) > 1:
            duplicate_guesses += 1
            # print("Duplicate guess detected.")

        # print("cumulative_guessed_letters:", cumulative_guessed_letters)
        flat_list = [
            item for sublist in cumulative_guessed_letters for item in sublist]
        # print("flat_list:", flat_list)
        # if len(flat_list) == 0:
        #     print(
        #         "Warning: No guesses were made or no guessed letters are available for analysis.")

    uncovered_progress = (maximum_word_length -
                          current_state.count('_')) / maximum_word_length
    critical_letter_uncover_rate = critical_letter_uncover_count / \
        len(critical_letters) if critical_letters else 0
    initial_letter_reveal = 1 if game_states[-1] and game_states[-1][0] != '_' else 0

    overall_success_rate = total_correct_guesses / \
        len(guesses) if guesses else 0

    # Ensure denominator is never 0
    late_game_half_length = max(len(guesses)//2, 1)
    late_game_success_rate = sum(
        guess_outcomes[-late_game_half_length:]) / late_game_half_length if guesses else 0

    final_state_achieved = 1 if '_' not in game_states[-1] else 0

    features = torch.tensor([
        uncovered_progress,
        missed_guesses,
        duplicate_guesses,
        len(guess_outcomes) - sum(guess_outcomes),  # Incorrect guesses
        len(cumulative_guessed_letters) / \
        maximum_word_length,  # Endgame proximity
        guess_diversity,
        initial_letter_reveal,
        critical_letter_uncover_rate,
        total_correct_guesses / \
        len(cumulative_guessed_letters) if cumulative_guessed_letters else 0,  # Guess efficiency
        overall_success_rate,  # Added overall success rate
        remaining_attempts,  # Added remaining attempts
        late_game_success_rate,  # Added late game success rate
        longest_success_streak,  # Added longest success streak
        final_state_achieved  # Added final state achieved
    ], dtype=torch.float)

    # # Debug prints
    # print(f"\nFeature Summary:")
    # print(f"Overall Success Rate: {overall_success_rate}")
    # print(f"Remaining Attempts: {remaining_attempts}")
    # print(f"Late Game Success Rate: {late_game_success_rate}")
    # print(f"Longest Success Streak: {longest_success_streak}")
    # print(
    #     f"Final State Achieved: {'True' if final_state_achieved else 'False'}")

    return features


def extract_char_features_from_state(word, char_frequency,
                                     max_word_length, ngram_n=3, normalize=True):
    """
        Extracts a variety of features from a single game state, including character frequencies,
        positional information, and n-gram features.

        Parameters:
        - state: The current game state (e.g., '_ppl_')
        - char_freq_dict: Dictionary mapping characters to their frequencies in some corpus
        - max_state_length: The maximum length of the game state to standardize feature length
        - ngram_size: The size of n-grams to consider for n-gram features
        - normalize: Whether to normalize certain features

        Returns:
        - A tensor of combined features extracted from the game state
    """

    word_len = len(word)

    # Encode the word
    encoded_word = encode_word(word)

    # print(f'encoded words: ', encode_word)

    # Features
    word_length_feature = [word_len / max_word_length] * word_len
    positional_feature = [pos / max_word_length for pos in range(word_len)]
    frequency_feature = [char_frequency.get(idx_to_char.get(
        char_idx, '_'), 0) for char_idx in encoded_word]

    # N-grams feature
    ngrams = extract_ngrams(word, ngram_n)
    ngram_feature = encode_ngrams(ngrams, ngram_n)

    # Truncate or pad ngram feature to match word length
    ngram_feature = ngram_feature[:word_len] + \
        [0] * (word_len - len(ngram_feature))

    # Normalizing features if required
    if normalize:
        max_freq = max(char_frequency.values()) if char_frequency else 1
        frequency_feature = [freq / max_freq for freq in frequency_feature]

        max_ngram_idx = max(char_to_idx.values())
        ngram_feature = [(ngram_idx / max_ngram_idx)
                         for ngram_idx in ngram_feature]

    # Combine the features
    combined_features = [
        torch.tensor(encoded_word, dtype=torch.long),
        torch.tensor(word_length_feature, dtype=torch.float),
        torch.tensor(positional_feature, dtype=torch.float),
        torch.tensor(frequency_feature, dtype=torch.float),
        torch.tensor(ngram_feature, dtype=torch.float)
    ]

    # Stack and pad/truncate to max_word_length
    features_stacked = torch.stack(combined_features, dim=1)

    # Pad or truncate the features to match max_word_length
    if word_len < max_word_length:
        padding = max_word_length - word_len
        features_padded = F.pad(
            features_stacked, (0, 0, 0, padding), "constant", 0)
    else:
        features_padded = features_stacked[:max_word_length, :]

    return features_padded  # Only return the feature tensor

# overall_success_rate, guess_outcomes = analyze_guess_outcomes(
#     game_states, guesses, maximum_word_length=None)


def process_game_state_features_and_missed_chars(word, char_frequency, max_word_length):
    # Process a single game state (word) to get features and missed characters
    feature_set = extract_char_features_from_state(
        word, char_frequency, max_word_length)  # Get features for the word

    # print(f"{feature_set.shape}")

    missed_chars = get_missed_characters(word)  # Get missed characters tensor
    # Return tensors for the single state
    return feature_set.squeeze(0), missed_chars


def process_game_sequence(game_states, guessed_letters_sequence,
                          char_frequency, max_word_length, max_seq_length):
    """
    Processes a sequence of game states along with the guessed letters to generate a comprehensive feature set
    for each state, capturing both the character-level features and game dynamics over the sequence.

    Parameters:
    - game_states: A list of strings, each representing a game state at a point in time.
    - guessed_letters_sequence: A list of letters guessed up to each point in the game sequence.
    - char_frequency: A dictionary mapping characters to their frequencies in a reference corpus.
    - max_word_length: The maximum length of the game state, used to standardize the size of feature tensors.
    - max_seq_length: The maximum number of game states to process, limiting the sequence length.

    Returns:
    - A tensor representing the sequence of combined features for each game state.
    - A tensor of missed characters for each game state in the sequence.
    """

    sequence_features = []  # To store combined features for each game state
    missed_chars_tensors = []  # To store missed characters for each game state

    # Ensure processing does not exceed the max sequence length
    for i, state in enumerate(game_states[:max_seq_length]):
        # Extract character features for the current state
        char_features, missed_chars = process_game_state_features_and_missed_chars(
            state, char_frequency, max_word_length)

        # Flatten the character features to concatenate with other features later
        flattened_char_features = char_features.view(-1)

        # Extract game dynamics features based on the cumulative guessed letters up to this state
        game_features = analyze_and_extract_features(
            game_states[:i+1], guessed_letters_sequence[:i+1], max_word_length)

        # Combine character-level and game-level features
        combined_features = torch.cat([flattened_char_features, game_features])

        # Store the combined features and missed characters for this state
        sequence_features.append(combined_features)
        missed_chars_tensors.append(missed_chars)

    # Convert lists to tensors
    sequence_tensor = torch.stack(sequence_features)
    missed_chars_tensor = torch.stack(missed_chars_tensors)

    # Verify the dimensions of the sequence tensor match expected dimensions
    expected_feature_length = flattened_char_features.shape[0] + \
        game_features.shape[0]
    assert sequence_tensor.shape[1] == expected_feature_length, \
        f"Feature length mismatch. Expected: {expected_feature_length}, Got: {sequence_tensor.shape[1]}"

    return sequence_tensor, missed_chars_tensor


def process_batch_of_games(guessed_states_batch, guessed_letters_batch,
                           char_frequency, max_word_length, max_seq_length):
    """
    Process a batch of game sequences to extract features for each game.
    """
    batch_features = []
    # This line assumes that missed characters are also part of the output.
    batch_missed_chars = []

    for game_states, guessed_letters in zip(guessed_states_batch, guessed_letters_batch):
        # Process each game sequence with all necessary parameters, including max_seq_length
        features, missed_chars = process_game_sequence(
            game_states, guessed_letters,
            char_frequency, max_word_length,
            max_seq_length
        )

        batch_features.append(features)
        # Collect missed chars for each game sequence
        batch_missed_chars.append(missed_chars)

    # Convert the list of tensors to a single tensor for the batch
    batch_features_tensor = torch.stack(batch_features)
    batch_missed_chars_tensor = torch.stack(batch_missed_chars)

    return batch_features_tensor, batch_missed_chars_tensor


def pad_and_reshape_labels(guesses, max_seq_length,
                           num_classes=len(char_to_idx)):
    """
    Pad, encode, and reshape labels for one-hot encoding.

    :param guesses: List of strings (guesses).
    :param max_seq_length: Maximum sequence length.
    :param char_to_idx: Dictionary mapping characters to indices.
    :param num_classes: Number of classes (size of the character set).
    :return: One-hot encoded labels of shape [batch_size, sequence_length, num_classes].
    """
    batch_size = len(guesses)

    # Initialize a zero tensor for padded labels
    padded_labels = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

    # Pad and encode each label in the batch
    for i, guess in enumerate(guesses):
        # Convert guess to indices using char_to_idx
        guess_indices = [char_to_idx.get(
            char, char_to_idx['']) for char in guess]

        # Pad the encoded guess
        length = min(len(guess_indices), max_seq_length)
        padded_labels[i, :length] = torch.tensor(guess_indices[:length])

    # Convert to one-hot encoding
    one_hot_labels = F.one_hot(padded_labels, num_classes=num_classes).float()

    return one_hot_labels
