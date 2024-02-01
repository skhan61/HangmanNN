import gc
import random
from collections import Counter

import numpy as np
import torch
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


def extract_hangman_game_features(current_game_state,
                                  cumulative_guessed_letters,
                                  maximum_word_length):
    # Assuming cumulative_guessed_letters is a list of lists
    # Access the first (and presumably only) list of guessed letters
    guessed_letters = cumulative_guessed_letters[0] \
        if cumulative_guessed_letters else [
    ]

    # Uncovered progress: fraction of the word that has been correctly guessed
    uncovered_progress = (maximum_word_length -
                          current_game_state.count('_')) / maximum_word_length

    # Missed guesses: number of guessed letters not in the word
    missed_guesses = len(
        [letter for letter in guessed_letters if letter not in current_game_state])

    # Duplicate guesses: number of letters guessed more than once
    duplicate_guesses = sum(guessed_letters.count(
        letter) > 1 for letter in set(guessed_letters))

    # Current guess accuracy: 1 if the latest guess is correct, 0 otherwise
    current_guess_accuracy = 1 if guessed_letters \
        and guessed_letters[-1] in current_game_state else 0

    # Endgame proximity: ratio of guesses made to maximum word length
    endgame_proximity = len(guessed_letters) / maximum_word_length

    # Guess diversity: number of unique guesses made divided by total number of guesses
    guess_diversity = len(set(guessed_letters)) / \
        len(guessed_letters) if guessed_letters else 0

    # Guess efficiency: ratio of unique correct guesses to total guesses
    correct_guesses = set(current_game_state.replace(
        '_', '')) & set(guessed_letters)
    guess_efficiency = len(correct_guesses) / \
        len(guessed_letters) if guessed_letters else 0

    # Initial letter reveal: 1 if the first letter is correctly guessed, 0 otherwise
    # initial_letter_reveal = 1 if current_game_state[0] != '_' else 0

    # def extract_hangman_game_features(current_game_state, cumulative_guessed_letters, \
    # maximum_word_length):
    # # Print the current game state for debugging
    # print(f"Current game state: '{current_game_state}'")
    # print(f"Cumulative guessed letters: {cumulative_guessed_letters}")

    # Your existing code for calculating features...
    guess_efficiency = len(correct_guesses) / \
        len(guessed_letters) if guessed_letters else 0

    # Add a print statement to debug the initial letter reveal logic
    if current_game_state:  # Check if the string is not empty
        # print(f"First character of current state: '{current_game_state[0]}'")
        initial_letter_reveal = 1 if current_game_state[0] != '_' else 0
    else:
        # print("Warning: Current game state is empty")
        initial_letter_reveal = 0  # Default value if current_game_state is empty

    # Critical letter uncover rate: rate at which vowels (critical letters) are uncovered
    critical_letters = 'aeiou'

    critical_letter_uncover_rate = sum(
        1 for letter in current_game_state if letter in critical_letters) \
        / len(critical_letters) if critical_letters else 0

    # Compile all features into a tensor
    features = torch.tensor([
        uncovered_progress,
        missed_guesses,
        duplicate_guesses,
        current_guess_accuracy,
        endgame_proximity,
        guess_diversity,
        initial_letter_reveal,
        critical_letter_uncover_rate,
        guess_efficiency
    ], dtype=torch.float)

    return features


def analyze_guess_outcomes(game_states, guesses, maximum_word_length=None):
    # Total number of new letters revealed by all guesses
    total_new_letters_revealed = 0
    guess_outcomes = []  # Tracks the success (1) or failure (0) of each guess

    for i in range(1, len(game_states)):
        prev_state = game_states[i-1]
        current_state = game_states[i]

        # # Debug print to check the lengths and contents of prev_state and current_state
        # print(
        #     f"Iteration {i}: prev_state='{prev_state}' (len={len(prev_state)}), \
        #         current_state='{current_state}' (len={len(current_state)})")

        new_letters_this_guess = 0
        for j in range(len(prev_state)):
            # Additional check to avoid index out of range
            if j < len(current_state) and prev_state[j] == '_' and current_state[j] != '_':
                new_letters_this_guess += 1

        # # Debug print to check new letters revealed by the current guess
        # print(
        #     f"New letters revealed in iteration {i}: {new_letters_this_guess}")

        total_new_letters_revealed += new_letters_this_guess

        # Determine the success of the guess: 1 if any new letters were revealed, 0 otherwise
        guess_success = int(new_letters_this_guess > 0)
        guess_outcomes.append(guess_success)

    if game_states:
        # Total number of positions revealed across all states
        total_revealed_positions = sum(
            len(state) - state.count('_') for state in game_states[1:])
        overall_success_rate = total_new_letters_revealed / \
            total_revealed_positions if total_revealed_positions else 0
    else:
        overall_success_rate = 0

    return overall_success_rate, guess_outcomes


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
    sequence_features = []
    missed_chars_tensors = []

    # Capture the original sequence length
    original_seq_len = len(game_states)

    for i, state in enumerate(game_states):
        if i >= max_seq_length:
            break

        # Extract character features for the current state
        char_features = extract_char_features_from_state(
            state, char_frequency, max_word_length)
        # Flatten the character features
        flattened_char_features = char_features.view(-1)

        # Extract game features for the current state
        cumulative_guessed_letters = guessed_letters_sequence[:i + 1]
        game_features = extract_hangman_game_features(
            state, cumulative_guessed_letters, max_word_length)

        # Combine flattened character features and game features
        combined_features = torch.cat([flattened_char_features, game_features])

        # Append the original sequence length as an additional feature
        combined_features_with_seq_len = torch.cat(
            [combined_features, torch.tensor([original_seq_len], dtype=torch.float)])

        sequence_features.append(combined_features_with_seq_len)

        # Extract and store missed characters for the current state
        # Assuming this function is defined
        missed_chars = get_missed_characters(state)
        missed_chars_tensors.append(missed_chars)

    # Stack all game state features to form the sequence tensor
    sequence_tensor = torch.stack(sequence_features)
    missed_chars_tensor = torch.stack(missed_chars_tensors)

    # Ensure the tensor shape matches the expected output dimensions
    expected_shape = (min(max_seq_length, len(game_states)),
                      flattened_char_features.shape[0] + game_features.shape[0] + 1)
    # +1 for the original sequence length

    assert sequence_tensor.shape == expected_shape, \
        f"Sequence tensor shape {sequence_tensor.shape} \
        does not match expected shape {expected_shape}"

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


# def process_game_sequence(game_states, guessed_letters_sequence,
#                           char_frequency, max_word_length, max_seq_length):
#     sequence_features = []
#     missed_chars_tensors = []

#     for i, state in enumerate(game_states):
#         if i >= max_seq_length:
#             break

#         # Extract character features for the current state
#         char_features = extract_char_features_from_state(
#             state, char_frequency, max_word_length)
#         # Flatten the character features
#         flattened_char_features = char_features.view(-1)

#         # Extract game features for the current state
#         cumulative_guessed_letters = guessed_letters_sequence[:i + 1]
#         game_features = extract_hangman_game_features(
#             state, cumulative_guessed_letters, max_word_length)

#         # Combine flattened character features and game features
#         combined_features = torch.cat([flattened_char_features, game_features])
#         sequence_features.append(combined_features)

#         # Extract and store missed characters for the current state
#         # Assuming this function is defined
#         missed_chars = get_missed_characters(state)
#         missed_chars_tensors.append(missed_chars)

#     # Stack all game state features to form the sequence tensor
#     sequence_tensor = torch.stack(sequence_features)
#     missed_chars_tensor = torch.stack(missed_chars_tensors)

#     # Ensure the tensor shape matches the expected output dimensions
#     expected_shape = (min(max_seq_length, len(game_states)),
#                       flattened_char_features.shape[0] + game_features.shape[0])

#     assert sequence_tensor.shape == expected_shape, \
#         f"Sequence tensor shape {sequence_tensor.shape} \
#         does not match expected shape {expected_shape}"

#     return sequence_tensor, missed_chars_tensor


# Note: The functions extract_char_features_from_state, extract_hangman_game_features, and get_missed_characters need to be defined.


# def process_game_sequence(game_states, guessed_letters_sequence,
#                           char_frequency, max_word_length, max_seq_length):

#     sequence_features = []
#     guess_outcomes = []

#     # Starting from the second state, determine the outcome of each guess
#     for i in range(1, len(game_states)):
#         # Guess is successful if the state changes
#         guess_success = game_states[i] != game_states[i-1]
#         guess_outcomes.append(int(guess_success))
#     guess_outcomes = [0] + guess_outcomes  # Default value for the first state

#     print(f"Guess Outcomes: {guess_outcomes}")

#     for i, state in enumerate(game_states):
#         if i >= max_seq_length:
#             break

#         print(f"\nProcessing State {i+1}/{len(game_states)}: '{state}'")

#         # Extract character features for the current state
#         char_features = extract_char_features_from_state(
#             state, char_frequency, max_word_length)
#         print(
#             f"Char Features Size: {char_features.size()}, Char Features: \n{char_features}")

#         # Extract game features for the current state
#         # Include all guesses up to the current state
#         cumulative_guessed_letters = guessed_letters_sequence[:i+1]

#         game_features = extract_hangman_game_features(
#             state, cumulative_guessed_letters, max_word_length)
#         print(f"Game Features: {game_features}")

#         # Repeat game features to match the size of char_features
#         game_features_repeated = game_features.repeat(char_features.size(0), 1)
#         print(f"Repeated Game Features Size: {game_features_repeated.size()}")

#         # Additional sequence-level features
#         additional_features = torch.tensor([guess_outcomes[i]], dtype=torch.float).unsqueeze(
#             0).repeat(char_features.size(0), 1)
#         print(
#             f"Additional Features Repeated Size: {additional_features.size()}")

#         # Combine all features
#         combined_features = torch.cat(
#             [char_features, game_features_repeated, additional_features], dim=1)
#         print(f"Combined Features Size: {combined_features.size()}")

#         sequence_features.append(combined_features)

#     # Stack all game state features to form the sequence tensor
#     sequence_tensor = torch.stack(sequence_features)
#     print(f"\nFinal Sequence Tensor Size: {sequence_tensor.size()}")

#     # Extract missed characters for each game state
#     missed_chars_tensor = torch.stack(
#         [get_missed_characters(state) for state in game_states])
#     print(f"Missed Chars Tensor Size: {missed_chars_tensor.size()}")

#     return sequence_tensor, missed_chars_tensor

#     print(f"Current game state: {current_game_state}")
#     print(
#         f"Cumulative guessed letters: {cumulative_guessed_letters}, Length: {len(cumulative_guessed_letters)}")

# # def extract_hangman_game_features(current_game_state,
# #                                   cumulative_guessed_letters, maximum_word_length):
#     """
#     Enhanced feature extraction for Hangman game analysis, including more game-level insights and
#     new aspects like letter positional information, game dynamics, and difficulty estimation.

#     Parameters:
#     - current_game_state (str): The current revealed state of the word being guessed, with '_'
#                 representing unrevealed letters.
#     - cumulative_guessed_letters (list): All letters guessed up to this point,
#                 including both correct and incorrect guesses.
#     - maximum_word_length (int): The maximum possible length of the word being guessed,
#                 used to normalize some of the features.

#     Returns:
#     - torch.Tensor: A tensor containing multiple features for game analysis:
#         - Uncovered Progress: The fraction of the word that has been correctly guessed so far.
#         - Missed/failed Guesses: The number of guessed letters that are not part of the word.
#         - Duplicate Guesses: The number of letters that have been guessed more than once.
#         - Current Guess Accuracy: 1 if the latest guess is correct (appears in the current game state), \
#             0 otherwise.
#         - Endgame Proximity: The ratio of the number of guesses made to the maximum word length,
#                     indicating how close the game is to ending based on the number of guesses.
#         - Guess Diversity: The diversity of guesses, calculated as the number of unique guesses
#                     made divided by the total number of guesses.
#         - Initial Letter Reveal: Proportion of initial letters correctly guessed.
#         - Critical Letter Uncover Rate: Rate at which critical letters are uncovered.
#         - Guess Efficiency: Ratio of unique correct guesses to total guesses.

#     Additional features like Recent Guess Trend and Guessed Letter Frequency Alignment could be considered
#     for future implementation to provide deeper insights into guessing patterns and alignment with common
#     language frequencies.
#     """

#     # Existing feature calculations
#     uncovered_progress = (maximum_word_length -
#                           current_game_state.count('_')) / maximum_word_length
#     missed_guesses = len(
#         [letter for letter in cumulative_guessed_letters if letter not in current_game_state])

#     duplicate_guesses = sum([1 for letter in set(
#         cumulative_guessed_letters) if cumulative_guessed_letters.count(letter) > 1])

#     current_guess_accuracy = 1 if cumulative_guessed_letters[-1] in current_game_state else 0

#     endgame_proximity = len(cumulative_guessed_letters) / maximum_word_length

#     guess_diversity = len(set(cumulative_guessed_letters)) / \
#         len(cumulative_guessed_letters) if cumulative_guessed_letters else 0

#     # New feature calculations
#     # Simple binary for initial letter reveal
#     initial_letter_reveal = 1 if current_game_state[0] != '_' else 0
#     # print(initial_letter_reveal)
#     critical_letters = 'aeiou'  # vowel
#     critical_letter_uncover_rate = sum(
#         [1 for letter in current_game_state if letter in critical_letters]) \
#         / len(critical_letters) if critical_letters else 0

#     correct_guesses = set(current_game_state.replace(
#         '_', '')) & set(cumulative_guessed_letters)
#     guess_efficiency = len(
#         correct_guesses) / len(cumulative_guessed_letters) \
#         if cumulative_guessed_letters else 0

#     # Compile all features into a tensor
#     features = torch.tensor([
#         uncovered_progress,
#         missed_guesses,
#         duplicate_guesses,
#         current_guess_accuracy,
#         endgame_proximity,
#         guess_diversity,
#         initial_letter_reveal,
#         critical_letter_uncover_rate,
#         guess_efficiency
#     ], dtype=torch.float)

#     return features


# def process_game_sequence(game_states, guessed_letters_sequence, char_frequency, max_word_length):
#     overall_success_rate, guess_outcomes = analyze_guess_outcomes(
#         game_states, guessed_letters_sequence)
#     sequence_features = []

#     for i, state in enumerate(game_states):
#         # Process each game state to extract character features
#         char_features = extract_char_features_from_state(
#             state, char_frequency, max_word_length)

#         # For each state, consider all guesses made up to that point
#         cumulative_guessed_letters = guessed_letters_sequence[:i]

#         # Extract game-level features based on the current state and cumulative guesses
#         game_features = extract_hangman_game_features(
#             state, cumulative_guessed_letters, max_word_length)

#         # Combine character features, game-level features, and outcome information
#         combined_features = torch.cat([
#             char_features,
#             game_features,
#             torch.tensor([overall_success_rate, guess_outcomes[i]
#                          if i < len(guess_outcomes) else 0], dtype=torch.float)
#         ])

#         sequence_features.append(combined_features)

#     # Stack all the combined features for each state to form the sequence tensor
#     return torch.stack(sequence_features)


# def process_batch_of_games(guessed_states_batch, guessed_letters_batch,
#                            char_frequency, max_word_length, max_seq_length):


# def process_game_sequence(game_states, guessed_letters_sequence,
#                           char_frequency, max_word_length, max_seq_length):
#     # Preprocess to get overall success rate and guess outcomes
#     overall_success_rate, guess_outcomes = analyze_guess_outcomes(
#         game_states, guessed_letters_sequence)

#     # Determine the number of features for each character in the game state
#     num_state_features = extract_char_features_from_state(
#         game_states[0], char_frequency, max_word_length).shape[-1]

#     # Dynamically determine the size for missed characters features
#     missed_chars_features_size = len(char_to_idx)

#     # Dynamically determine the size for game features
#     game_features_size = extract_hangman_game_features(
#         game_states[0], guessed_letters_sequence[0], max_word_length).numel()

#     # Adjust the size to include state features, missed characters features, game features, \
#     # overall success rate, and guess outcomes for each timestep
#     combined_sequence_features_size = max_word_length * \
#         num_state_features + missed_chars_features_size + \
#         game_features_size + 1 + \
#         1  # +1 for overall success rate, +1 for guess outcome per timestep

#     # Initialize tensors for the combined sequence features and missed characters for each game state
#     combined_sequence_features = torch.zeros(
#         max_seq_length, combined_sequence_features_size)
#     sequence_missed_chars = torch.zeros(
#         max_seq_length, missed_chars_features_size)

#     for i, (current_state, guessed_letters) in enumerate(zip(game_states, guessed_letters_sequence)):
#         if i < max_seq_length:
#             # Extract features from the current state and the guessed letters
#             state_features, current_missed_chars = process_game_state_features_and_missed_chars(
#                 current_state, char_frequency, max_word_length)
#             guessed_letters_features = torch.tensor(
#                 encode_guessed_letters(guessed_letters, char_to_idx))

#             # Extract the additional game-related features
#             game_features = extract_hangman_game_features(
#                 current_state, guessed_letters, max_word_length)

#             # Include the overall success rate and the guess outcome for the current timestep
#             # Use 0 for the first state where no guess was made
#             current_guess_outcome = torch.tensor(
#                 [guess_outcomes[i-1]]) if i > 0 else torch.tensor([0])

#             # Combine the state features, guessed letters features, game-related features, \
#             # overall success rate, and guess outcome for the current timestep
#             combined_features = torch.cat([state_features.view(-1), guessed_letters_features,
#                                           game_features, torch.tensor(
#                                               [overall_success_rate]),
#                                            current_guess_outcome])

#             combined_sequence_features[i] = combined_features
#             sequence_missed_chars[i] = current_missed_chars

#     return combined_sequence_features, sequence_missed_chars


# def process_batch_of_games(guessed_states_batch, guessed_letters_batch,
#                            char_frequency, max_word_length, max_seq_length):

#     batch_size = len(guessed_states_batch)  # Number of games in the batch

#     # Assuming process_game_sequence returns tensors of correct shape
#     # We need to know the shape of the tensor it returns to initialize batch_features correctly
#     sample_features, _ = process_game_sequence(
#         guessed_states_batch[0], guessed_letters_batch[0], char_frequency,
#         max_word_length, max_seq_length
#     )
#     feature_shape = sample_features.shape[-1]

#     # Initialize tensors for the entire batch
#     batch_features = torch.zeros(batch_size, max_seq_length, feature_shape)
#     batch_missed_chars = torch.zeros(
#         batch_size, max_seq_length, len(char_to_idx))

#     # Process each game in the batch
#     for i in range(batch_size):
#         game_states = guessed_states_batch[i]
#         guessed_letters = guessed_letters_batch[i]

#         sequence_features, sequence_missed_chars = process_game_sequence(
#             game_states, guessed_letters, char_frequency, max_word_length, max_seq_length
#         )

#         # Ensure the size of sequence_features matches the expected size in batch_features
#         # If there's a mismatch, you might need to adjust process_game_sequence
#         batch_features[i] = sequence_features
#         batch_missed_chars[i] = sequence_missed_chars

#     return batch_features, batch_missed_chars

# def process_single_game_state(game_state,
#                               char_frequency,
#                               max_word_length):
#     print("Game state received:", game_state)  # Debugging statement
#     current_state, guessed_characters = game_state[0], game_state[1]

#     # Process this single game state
#     sequence_features, sequence_missed_chars = process_game_sequence(
#         [current_state], char_frequency, max_word_length, 1)  # max_seq_length is 1 for single game state

#     # Since it's a single game state, we extract the first element from the batch
#     return sequence_features[0], sequence_missed_chars[0]


# # Dummy batch of game states similar to 'e__e__e'
# guessed_states_batch = [
#     ['e__e__', 'e_e_e_', 'ee_e__', 'eee_e_', 'eeeee_'],
#     ['_e__e_', '__e_e_', '_ee__e', '_eee_e', '_eeee_'],
#     ['e___e_', 'e__ee_', 'e_e_e_', 'ee__e_', 'eee_e_']
# ]

# # Dummy batch of guessed letters for each state
# guessed_letters_batch = [
#     ['a', 'b', 'c', 'd', 'e'],
#     ['f', 'g', 'h', 'i', 'j'],
#     ['k', 'l', 'm', 'n', 'o']
# ]


# # Dummy batch of game states similar to 'e__e__e'
# guessed_states_batch = [
#     ['e__e__'],
# ]

# # Dummy batch of guessed letters for each state
# guessed_letters_batch = [
#     ['a'],
# ]


# state_fets, state_miss_char = process_game_sequence(game_states, char_frequency,
#                           max_word_length, max_seq_length)
# print(f'state fets shape: ', state_fets.shape)
# print(f'state_miss_chars shape: ', state_miss_char.shape)
# print()
# # Process the dummy batch
# batch_features, batch_missed_chars = process_batch_of_games(
#     guessed_states_batch, guessed_letters_batch, char_frequency, max_word_length, max_seq_length)
# # Outputs
# print("Batch Features Shape:", batch_features.shape)  # Expected: [3, 5, num_features]
# print("Batch Missed Chars Shape:", batch_missed_chars.shape)  # Expected: [3, 5, len(char_to_idx)]
gc.collect()


# def process_game_sequence(game_states, guessed_letters_sequence,
#                           char_frequency, max_word_length, max_seq_length):
#     # Determine the number of features for each character in the game state
#     num_state_features = extract_char_features_from_state(
#         game_states[0], char_frequency, max_word_length).shape[-1]
#     # print(f"Number of state features per character: {num_state_features}")  # Debugging

#     # Initialize tensors for the combined sequence features and missed characters for each game state
#     combined_sequence_features = torch.zeros(
#         max_seq_length, max_word_length * num_state_features + 28)
#     sequence_missed_chars = torch.zeros(max_seq_length, len(char_to_idx))

#     # print(f"Initialized combined sequence features shape: {combined_sequence_features.shape}")  # Debugging
#     # print(f"Initialized sequence missed chars shape: {sequence_missed_chars.shape}")  # Debugging

#     for i, (current_state, guessed_letters) in enumerate(zip(game_states, guessed_letters_sequence)):
#         if i < max_seq_length:
#             # print(f"\nProcessing game state {i}: {current_state}")  # Debugging
#             # print(f"Guessed letters at this state: {guessed_letters}")  # Debugging

#             # Extract features from the current state and the guessed letters
#             state_features, current_missed_chars = process_game_state_features_and_missed_chars(
#                 current_state, char_frequency, max_word_length)

#             guessed_letters_features = torch.tensor(
#                 encode_guessed_letters(guessed_letters, char_to_idx))

#             # Call the function with the sample data
#             game_features = extract_hangman_game_features(current_state,
#                                                           guessed_letters, max_word_length)

#             # Print the extracted game features for inspection
#             print("Extracted Game Features:", game_features)

#             # Debugging
#             print(f"Current state features shape: {state_features.shape}")
#             # Debugging
#             print(f"Current missed chars shape: {current_missed_chars.shape}")
#             # Debugging
#             print(
#                 f"Guessed letters features shape: {guessed_letters_features.shape}")

#             # Combine the state features and guessed letters features
#             combined_features = torch.cat(
#                 [state_features.view(-1), guessed_letters_features])
#             print(f"Combined features shape for this state: {combined_features.shape}")  # Debugging

#             combined_sequence_features[i] = combined_features
#             sequence_missed_chars[i] = current_missed_chars

#             print(f"Updated combined sequence features shape: {combined_sequence_features.shape}")  # Debugging
#             print(f"Updated sequence missed chars shape: {sequence_missed_chars.shape}")  # Debugging

#     # Return the tensors for the entire sequence of game states
#     return combined_sequence_features, sequence_missed_chars
