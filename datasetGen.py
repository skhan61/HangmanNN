import sys
from pathlib import Path
import warnings

import warnings
import pandas as pd
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

import sys
# Custom library paths
sys.path.extend(['../', './scr'])

from scr.utils import set_seed
from scr.utils import read_words

set_seed(42)

import torch
import torch.nn as nn

torch.set_float32_matmul_precision('medium')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scr.feature_engineering import build_feature_set, \
    process_single_word_inference, calculate_char_frequencies
from scr.utils import *

import random
from collections import Counter

# MASK_PROB = 0.5

# Limit the number of words to a smaller number for debugging
word_list = read_words('/home/sayem/Desktop/Hangman/words_250000_train.txt', limit=None)
# word_list = word_list[:100]
# # # Randomly select 1000 words
# # unseen_words = random.sample(word_list, 1000)c

def calculate_word_frequencies(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}

import random

# ..
word_frequencies = calculate_word_frequencies(word_list)
char_frequency = calculate_char_frequencies(word_list)
max_word_length = max(len(word) for word in word_list)


from scr.feature_engineering import build_feature_set, \
    process_single_word_inference, calculate_char_frequencies
from scr.utils import *

import random
from collections import Counter

# MASK_PROB = 0.5

# Limit the number of words to a smaller number for debugging
word_list = read_words('/home/sayem/Desktop/Hangman/words_250000_train.txt', limit=None)
# word_list = word_list[:100]
# # # Randomly select 1000 words
# # unseen_words = random.sample(word_list, 1000)c

def calculate_word_frequencies(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}

import random

import scr.utils import calculate_word_frequencies

# ..
word_frequencies = calculate_word_frequencies(word_list)
char_frequency = calculate_char_frequencies(word_list)
max_word_length = max(len(word) for word in word_list)