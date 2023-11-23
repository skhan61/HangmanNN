import gc
import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler

MAX_EXTRA_WEIGHT = 100  # Example value, adjust as needed
TARGET_WIN_RATE = 0.5   # This is an example value for scaling the weight
MAX_INDICES = 10000     # Example value, adjust as needed


class PerformanceBasedSampler(Sampler):
    def __init__(self, data_source, performance_metrics):
        # print("Initializing PerformanceBasedSampler")
        self.data_source = data_source
        self.performance_metrics = performance_metrics
        self.indices = self.generate_indices()
        # print(f"Initial indices generated: {len(self.indices)}")

    def generate_indices(self):
        epsilon = 1e-6  # A small value to avoid division by zero
        indices = []
        extra_weights_distribution = {}  # For debugging

        for idx, data in enumerate(self.data_source):
            word_length = len(data[3])
            performance = self.performance_metrics.get(word_length, {})
            win_rate = performance.get('win_rate', 1)

            # Calculate extra weight based on win rate
            extra_weight = max(1, min(int(TARGET_WIN_RATE
                                          / (win_rate + epsilon)), MAX_EXTRA_WEIGHT))
            indices.extend([idx] * extra_weight)

            # Collect data for debugging
            extra_weights_distribution.setdefault(extra_weight, 0)
            extra_weights_distribution[extra_weight] += 1

            # Debug line for every 100 entries
            # if idx % 100 == 0:
            # print(f"Processed {idx} entries, current index count: {len(indices)}")

        # Limit the total number of indices
        if len(indices) > MAX_INDICES:
            indices = indices[:MAX_INDICES]

        # print(f"Total indices generated: {len(indices)}")
        # print(f"Extra weights distribution: {extra_weights_distribution}")
        return indices

    def __iter__(self):
        # print("Creating iterator for indices")
        return iter(self.indices)

    def __len__(self):
        # print(f"Length of sampler: {len(self.indices)}")
        return len(self.indices)


def update_sampler(sampler, new_performance_metrics):
    # print("Updating sampler with new performance metrics")
    sampler.performance_metrics = new_performance_metrics
    sampler.indices = sampler.generate_indices()
    # print(f"Updated indices count: {len(sampler.indices)}")


class PerformanceBasedSampler(Sampler):
    def __init__(self, data_source, performance_metrics):
        self.data_source = data_source
        self.performance_metrics = performance_metrics
        self.indices = self.generate_indices()

    def generate_indices(self):
        epsilon = 1e-6
        indices = []

        for idx, data in enumerate(self.data_source):
            word_length = len(data['word'])
            difficulty = data['difficulty']
            outcome = data['outcome']
            performance = self.performance_metrics.get(
                word_length, {}).get(difficulty, {}).get(outcome, {})
            win_rate = performance.get('win_rate', 1)

            extra_weight = max(
                1, min(int(TARGET_WIN_RATE / (win_rate + epsilon)), MAX_EXTRA_WEIGHT))
            indices.extend([idx] * extra_weight)

        if len(indices) > MAX_INDICES:
            random.shuffle(indices)
            indices = indices[:MAX_INDICES]

        return indices

    # ... [rest of the class remains the same] ...

# Update the sampler with new performance metrics when needed


def update_sampler(sampler, new_performance_metrics):
    sampler.performance_metrics = new_performance_metrics
    sampler.indices = sampler.generate_indices()


# dont change below
def stratified_sample_by_length_and_frequency(word_list, word_freq_dict,
                                              num_stratified_samples):

    word_groups = group_words_by_length(word_list)
    total_words = sum(len(group) for group in word_groups.values())
    stratified_sampled_words = []

    for word_length, words in word_groups.items():
        words_sorted_by_freq = sorted(
            words, key=lambda w: word_freq_dict.get(w, 0), reverse=True)

        # Calculate the number of samples to draw from this group
        num_samples_per_group = max(
            1, round(num_stratified_samples * len(words) / total_words))

        # Split the sorted words into high and low frequency halves
        split_index = len(words) // 2
        high_freq_words = words_sorted_by_freq[:split_index]
        low_freq_words = words_sorted_by_freq[split_index:]

        # Determine the number of samples from high and low frequency words
        high_freq_samples = num_samples_per_group // 2
        low_freq_samples = num_samples_per_group - high_freq_samples

        # Sample words from both high and low frequency subsets
        sampled_high_freq_words = random.sample(
            high_freq_words, min(high_freq_samples, len(high_freq_words)))
        sampled_low_freq_words = random.sample(
            low_freq_words, min(low_freq_samples, len(low_freq_words)))

        stratified_sampled_words.extend(
            sampled_high_freq_words + sampled_low_freq_words)

    return stratified_sampled_words

# Helper function to group words by their length


def group_words_by_length(word_list):
    word_groups = {}
    for word in word_list:
        word_length = len(word)
        if word_length not in word_groups:
            word_groups[word_length] = []
        word_groups[word_length].append(word)
    return word_groups


gc.collect()
