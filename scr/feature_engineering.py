import torch
import random
from collections import Counter
import random
import gc
import numpy as np

# 1. Common Utility Functions
# Character set and mapping
char_to_idx = {char: idx for idx, char in enumerate(['_'] + list('abcdefghijklmnopqrstuvwxyz'))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_word(word):
    return [char_to_idx[char] for char in word]

def get_missed_characters(word):
    all_chars = set(char_to_idx.keys())
    present_chars = set(word)
    missed_chars = all_chars - present_chars
    return torch.tensor([1 if char in missed_chars else 0 for char in char_to_idx], dtype=torch.float)

def calculate_char_frequencies(word_list):
    char_counts = Counter(''.join(word_list))
    return {char: char_counts[char] / sum(char_counts.values()) for char in char_to_idx}

def extract_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]

    encoded_ngrams = [char_to_idx[char] for ngram in ngrams for char in ngram]
    fixed_length = n * 2
    return encoded_ngrams[:fixed_length] + [0] * (fixed_length - len(encoded_ngrams))

def pad_tensor(tensor, length):
    return torch.cat([tensor, torch.zeros(length - len(tensor))], dim=0)

# 2. Training Data Preparation

# def generate_masked_input_and_labels(encoded_word, mask_prob=0.8):
#     return [(char_to_idx['_'], char_idx) \
#         if random.random() < mask_prob else \
#             (char_idx, 0) \
#         for char_idx in encoded_word]

def generate_masked_input_and_labels(encoded_word, mask_prob=0.8):
    # Generate a random array of the same length as encoded_word
    random_values = np.random.rand(len(encoded_word))

    # Create an array of '_' indices (assuming '_' is mapped to 0 in char_to_idx)
    masked_indices = np.full(len(encoded_word), char_to_idx['_'])

    # Convert encoded_word to a numpy array for efficient operations
    encoded_array = np.array(encoded_word)

    # Apply the mask where random values are less than mask_prob
    # If random value is less than mask_prob, use masked index ('_'), else use the original character index
    result = np.where(random_values < mask_prob, masked_indices, encoded_array)

    # Generate labels (0 for unmasked, original index for masked)
    labels = np.where(result == masked_indices, encoded_array, 0)

    # Convert result and labels back to list of tuples
    return list(zip(result, labels))

def encode_ngrams(ngrams, n):
    encoded_ngrams = [char_to_idx[char] for ngram in ngrams for char in ngram]
    fixed_length = n * 2
    return encoded_ngrams[:fixed_length] + [0] * (fixed_length - len(encoded_ngrams))


def calculate_word_frequencies(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}


# 3. Train/Inference Data Preparation

def add_features_for_training(word, char_frequency, max_word_length, \
    mask_prob=0.5, ngram_n=2, normalize=True):
    encoded_word = encode_word(word)

    # Masked input and labels generation
    masked_input, label = zip(*generate_masked_input_and_labels(encoded_word, mask_prob))

    # Build feature set
    feature_set = build_feature_set(word, char_frequency, \
        max_word_length, masked_input, ngram_n, normalize=True)

    missed_chars = get_missed_characters(word)

    return feature_set, torch.tensor(label, \
        dtype=torch.float), missed_chars

# 3. Inference Data Preparation

def process_single_word_inference(word, char_frequency, \
    max_word_length, ngram_n=2, normalize=True):
    encoded_word = encode_word(word)
    feature_set = build_feature_set(word, char_frequency, max_word_length, \
        encoded_word, ngram_n, normalize=True)
    missed_chars = get_missed_characters(word)
    return feature_set, missed_chars.unsqueeze(0)

# def process_single_word_inference(word, \
#     max_word_length, ngram_n=2, normalize=True):
#     encoded_word = encode_word(word)
#     feature_set = build_feature_set(word, max_word_length, \
#         encoded_word, ngram_n, normalize=True)
#     missed_chars = get_missed_characters(word)
#     return feature_set, missed_chars.unsqueeze(0)

# 4. Shared Feature Set Construction
def build_feature_set(word, char_frequency, max_word_length, \
    input_sequence, ngram_n, normalize):
    word_len = len(word)
    
    word_length_feature = [word_len / max_word_length] * word_len
    positional_feature = list(range(word_len))
    frequency_feature = [char_frequency.get(idx_to_char.get(char_idx, 0), 0) \
        for char_idx in input_sequence]

    ngrams = extract_ngrams(word, ngram_n)
    ngram_feature = encode_ngrams(ngrams, ngram_n)

    # Truncate or pad ngram feature to match word length
    ngram_feature = ngram_feature[:word_len] + [0] * (word_len - len(ngram_feature))

    # Normalizing features if required
    if normalize:
        positional_feature = [pos / max_word_length for pos in positional_feature]
        max_freq = max(char_frequency.values()) if char_frequency else 1
        frequency_feature = [freq / max_freq for freq in frequency_feature]
    
        # Normalizing ngram features
        max_ngram_idx = max(char_to_idx.values())  # Assuming char_to_idx is accessible here
        ngram_feature = [(ngram_idx / max_ngram_idx) for ngram_idx in ngram_feature]

    features = [
        torch.tensor(input_sequence, dtype=torch.long),
        torch.tensor(word_length_feature, dtype=torch.float),
        torch.tensor(positional_feature, dtype=torch.float),
        # torch.tensor(frequency_feature, dtype=torch.float),
        # torch.tensor(ngram_feature, dtype=torch.float)
    ]

    return torch.stack(features, dim=1)

gc.collect()



