import torch
import random
from collections import Counter

# 1. Common Utility Functions
# Character set and mapping
char_to_idx = {char: idx for idx, char in enumerate(['_'] + list('abcdefghijklmnopqrstuvwxyz'))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

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

def generate_masked_input_and_labels(encoded_word, mask_prob=0.5):
    return [(char_to_idx['_'], char_idx) if random.random() < mask_prob else (char_idx, 0) \
        for char_idx in encoded_word]

def add_features_for_training(word, char_frequency, max_word_length, \
    mask_prob=0.5, ngram_n=2):
    encoded_word = encode_word(word)
    masked_input, label = zip(*generate_masked_input_and_labels(encoded_word, mask_prob))
    feature_set = build_feature_set(word, char_frequency, max_word_length, masked_input, ngram_n)
    missed_chars = get_missed_characters(word)  # This line is missing in your current implementation
    return feature_set, torch.tensor(label, dtype=torch.float), missed_chars

# 3. Inference Data Preparation

def process_single_word_inference(word, char_frequency, max_word_length, ngram_n=2):
    encoded_word = encode_word(word)
    feature_set = build_feature_set(word, char_frequency, max_word_length, encoded_word, ngram_n)
    missed_chars = get_missed_characters(word)
    return feature_set, missed_chars.unsqueeze(0)

# 4. Shared Feature Set Construction

def build_feature_set(word, char_frequency, max_word_length, input_sequence, ngram_n):
    word_length_feature = [len(word) / max_word_length] * len(word)
    positional_feature = list(range(len(word)))
    frequency_feature = [char_frequency.get(idx_to_char[char_idx], 0) for char_idx in input_sequence]
    ngrams = extract_ngrams(word, ngram_n)
    ngram_feature = encode_ngrams(ngrams, ngram_n)

    # Convert and pad features
    features = [torch.tensor(f, dtype=torch.float if isinstance(f[0], float) else torch.long) for f in [input_sequence, word_length_feature, positional_feature, frequency_feature, ngram_feature]]
    max_length = max(len(f) for f in features)
    padded_features = [pad_tensor(f, max_length) for f in features]
    return torch.stack(padded_features, dim=1)


