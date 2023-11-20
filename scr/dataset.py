import random
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset
from collections import defaultdict
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

import gc

gc.collect()



# def process_word(word):
#     word_length = len(word)
#     max_variants = min(word_length, 15)  # Increase to a higher number
#     max_masks = word_length - 1  # Allow up to almost full masking

#     return generate_masked_word_variants(word, max_variants, max_masks)

# def generate_masked_word_variants(word, max_variants, max_masks):
#     word_length = len(word)
#     indices = list(range(word_length))
#     masked_versions = set()

#     while len(masked_versions) < max_variants:
#         num_masks = random.randint(1, max_masks)  # Vary the number of masks
#         mask_indices = set(random.sample(indices, num_masks))
#         masked_word = ''.join(c if i not in mask_indices else '_' for i, c in enumerate(word))
#         masked_versions.add(masked_word)

#     return list(masked_versions)

# def process_word(word):
#     word_length = len(word)
#     max_variants = min(word_length, 10)  # Increase to a higher number
#     max_masks = word_length - 1  # Allow up to almost full masking

#     return generate_masked_word_variants(word, max_variants, max_masks)

class HangmanDataset(Dataset):
    def __init__(self, word_list, char_freq_dict, max_word_length, \
        mask_probability, ngram_size, num_parallel_workers=5, processing_timeout=10):

        self.char_freq_dict = char_freq_dict
        self.max_word_length = max_word_length
        self.mask_probability = mask_probability
        self.ngram_size = ngram_size
        self.num_parallel_workers = num_parallel_workers
        self.processing_timeout = processing_timeout

        # Process the provided word list directly
        self.data = self.process_data_parallel(word_list)

    def process_data_parallel(self, word_list):
        chunk_size = 100  # Adjust based on memory constraints
        self.data = []

        # Create a persistent pool for parallel processing
        with Pool(self.num_parallel_workers) as pool:
            results = [pool.map_async(self.process_single_word, word_list[i:i+chunk_size])
                       for i in range(0, len(word_list), chunk_size)]

            # Collect results as they complete
            for result in results:
                self.data.extend([item for sublist in result.get() for item in sublist])

        return self.data

    def process_single_word(self, word):
        processed_data = []
        masked_variants = process_word(word)
        for masked_word in masked_variants:
            feature_set, label_tensor, missed_chars = add_features_for_training(
                masked_word, self.char_freq_dict, self.max_word_length, 
                self.mask_probability, self.ngram_size, normalize=True)

            # Share memory for tensors
            feature_set = feature_set.clone().detach().share_memory_()
            label_tensor = label_tensor.clone().detach().share_memory_()
            missed_chars = missed_chars.clone().detach().share_memory_()

            processed_data.append((feature_set, label_tensor, missed_chars, word))

        return processed_data

    def __getitem__(self, index):
        feature_set, label_tensor, missed_chars, original_word = self.data[index]
        return feature_set, label_tensor, missed_chars, original_word

    def __len__(self):
        return len(self.data)

from torch.utils.data import DataLoader



class ProcessedHangmanDataset(Dataset):
    def __init__(self, pkl_files_directory):
        self.data = []
        for file_path in pkl_files_directory.glob("*.pkl"):
            with open(file_path, 'rb') as file:
                game_states = pickle.load(file)
                for state in game_states:
                    feature_set, label_tensor, _ = add_features_for_training(
                        state[0], char_freq_dict, max_word_length, 
                        mask_probability, ngram_size, normalize=True)
                    self.data.append((feature_set, label_tensor))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# # Load the dataset
# processed_dataset = ProcessedHangmanDataset(pkls_dir)

# # Create DataLoader for batching
# data_loader = DataLoader(processed_dataset, batch_size=32, shuffle=True)

# # Training Loop Example
# for epoch in range(num_epochs):
#     for features, labels in data_loader:
#         # Training steps
#         # Pass features and labels to your model



# class HangmanDataset(Dataset):
#     def __init__(self, word_list, char_freq_dict, max_word_length, \
#         mask_probability, ngram_size, 
#                  word_freq_dict, num_stratified_samples=2000, \
#                     num_parallel_workers=5, processing_timeout=10):
        
#         self.char_freq_dict = char_freq_dict
#         self.max_word_length = max_word_length
#         self.mask_probability = mask_probability
#         self.ngram_size = ngram_size
#         self.num_parallel_workers = num_parallel_workers
#         self.processing_timeout = processing_timeout

#         stratified_words = self.stratified_sample_by_length_and_frequency(word_list, \
#             word_freq_dict, num_stratified_samples)
#         self.data = self.process_data_parallel(stratified_words)


    # def process_data_parallel(self, word_list):
    #     with Pool(self.num_parallel_workers) as pool:
    #         results = pool.map(self.process_single_word, word_list)
    #     self.data = [item for sublist in results for item in sublist]
        
    #     return self.data

    # def process_data_parallel(self, word_list):
    #         chunk_size = 100  # Adjust this based on your memory constraints
    #         self.data = []
            
    #         # Create a persistent pool
    #         with Pool(self.num_parallel_workers) as pool:
    #             # Process data in chunks asynchronously
    #             results = [pool.map_async(self.process_single_word, word_list[i:i+chunk_size])
    #                     for i in range(0, len(word_list), chunk_size)]

    #             # Collect results as they complete
    #             for result in results:
    #                 self.data.extend([item for sublist in result.get() for item in sublist])

    #         return self.data

    # def process_single_word(self, word):
    #     processed_data = []
    #     masked_variants = process_word(word)
    #     for masked_word in masked_variants:
    #         feature_set, label_tensor, missed_chars = add_features_for_training(
    #             masked_word, self.char_freq_dict, self.max_word_length, 
    #             self.mask_probability, self.ngram_size, normalize=True)

    #         # Share memory for tensors
    #         feature_set = feature_set.clone().detach().share_memory_()
    #         label_tensor = label_tensor.clone().detach().share_memory_()
    #         missed_chars = missed_chars.clone().detach().share_memory_()

    #         processed_data.append((feature_set, label_tensor, missed_chars, word))
    
    #     return processed_data

    # def process_single_word(self, word):
    #     processed_data = []
    #     masked_variants = process_word(word)
    #     for masked_word in masked_variants:
    #         feature_set, label_tensor, missed_chars = add_features_for_training(
    #             masked_word, self.char_freq_dict, self.max_word_length, 
    #             self.mask_probability, self.ngram_size, normalize=True)
    #         processed_data.append((feature_set, label_tensor, missed_chars, word))
    
    #     return processed_data

    # def stratified_sample_by_length_and_frequency(self, word_list, word_freq_dict, \
    #     num_stratified_samples):
    #     word_groups = self.group_words_by_length(word_list)
    #     total_words = sum(len(group) for group in word_groups.values())
    #     stratified_sampled_words = []

    #     for word_length, words in word_groups.items():
    #         words_sorted_by_freq = sorted(words, key=lambda w: word_freq_dict.get(w, 0))
    #         half_len = len(words) // 2
    #         high_freq_words = words_sorted_by_freq[:half_len]
    #         low_freq_words = words_sorted_by_freq[half_len:]

    #         num_samples_per_group = int(num_stratified_samples * len(words) / total_words)
    #         high_freq_samples = min(len(high_freq_words), num_samples_per_group // 2)
    #         low_freq_samples = min(len(low_freq_words), num_samples_per_group // 2)

    #         stratified_sampled_words.extend(random.sample(high_freq_words, high_freq_samples))
    #         stratified_sampled_words.extend(random.sample(low_freq_words, low_freq_samples))

    #     return stratified_sampled_words

    # def group_words_by_length(self, word_list):
    #     word_groups_by_length = defaultdict(list)
    #     for word in word_list:
    #         word_groups_by_length[len(word)].append(word)
    #     return word_groups_by_length

    # def __getitem__(self, index):
    #     # Ensure this returns a tuple with exactly four elements
    #     feature_set, label_tensor, missed_chars, original_word = self.data[index]
    #     return feature_set, label_tensor, missed_chars, original_word

    # def __len__(self):
    #     return len(self.data)

    # def __len__(self):
    #     return len(self.data)



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
            # print(feature)
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



    # def generate_all_masked_versions(self, word):
    #     masked_versions = []
    #     for i in range(1 << len(word)):  # 2^len(word) combinations
    #         masked_word = ''.join(word[j] if i & (1 << j) else '_' for j in range(len(word)))
    #         masked_versions.append(masked_word)
    #     return masked_versions
    
    # def generate_all_masked_versions(self, word, num_samples=5):
    #     masked_versions = []
    #     for _ in range(num_samples):  # Generate a limited number of samples
    #         masked_word = ''.join(c if random.random() > self.mask_prob else '_' for c in word)
    #         masked_versions.append(masked_word)
    #     return masked_versions
