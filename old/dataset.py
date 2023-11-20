import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import torch
from scr.feature_engineering import add_features_for_training, \
    calculate_char_frequencies, process_single_word_inference, char_to_idx,idx_to_char 
import random

from torch.utils.data import Dataset

class HangmanDataset(Dataset):
    def __init__(self, word_list, char_frequency, \
        max_word_length, mask_prob, ngram_n):
        self.word_list = word_list
        self.char_frequency = char_frequency
        self.max_word_length = max_word_length
        self.mask_prob = mask_prob
        self.ngram_n = ngram_n

        self.data = self._process_data()

    def _process_data(self):
        data = []
        for word in self.word_list:
            for masked_word in self.generate_all_masked_versions(word):
                print(masked_word)  # Optional, for debugging purposes

                # Call add_features_for_training for each masked word
                feature_set, label_tensor, missed_chars = add_features_for_training(
                    masked_word, self.char_frequency, self.max_word_length,
                    self.mask_prob, self.ngram_n, normalize=True)

                # Append the processed data
                data.append((feature_set, label_tensor, missed_chars, masked_word))

        return data


    def generate_all_masked_versions(self, word):
        masked_versions = []
        for i in range(1 << len(word)):  # 2^len(word) combinations
            masked_word = ''.join(word[j] if i & (1 << j) else '_' for j in range(len(word)))
            masked_versions.append(masked_word)
        return masked_versions

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        


def collate_fn(batch):
    batch_features, batch_labels, \
        batch_missed_chars, original_words = zip(*batch)

    max_length = max(feature.size(0) for feature in batch_features)

    padded_features = []
    padded_labels = []

    for feature, label in zip(batch_features, batch_labels):
        # print(f"Feature tensor shape before padding: {feature.shape}")

        # Check if feature tensor has the expected number of dimensions
        if feature.dim() == 2 and feature.size(1) > 0:
            padded_feature = F.pad(feature, (0, 0, 0, \
                max_length - feature.size(0)))
            # print('okay')
        else:
            # Handle tensors that do not have the expected shape
            # Example: Skip this feature or apply a different padding logic
            print(feature)
            continue

        padded_label = F.pad(label, (0, max_length - label.size(0)))

        padded_features.append(padded_feature)
        padded_labels.append(padded_label)

    # Convert list of tensors to tensors with an added batch dimension
    padded_features = torch.stack(padded_features, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    batch_missed_chars = torch.stack(batch_missed_chars, dim=0)

    lengths_features = torch.tensor([feature.size(0) for feature in batch_features], dtype=torch.long)

    return padded_features, padded_labels, batch_missed_chars, lengths_features, original_words

