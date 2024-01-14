import gc
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Sampler

MAX_EXTRA_WEIGHT = 100  # Example value, adjust as needed
TARGET_WIN_RATE = 50   # This is an example value for scaling the weight
MAX_INDICES = 32     # Example value, adjust as needed


class PerformanceBasedSampler(Sampler):
    def __init__(self, data_source, performance_metrics, max_word_length=29,
                 target_win_rate=TARGET_WIN_RATE, max_weight=MAX_EXTRA_WEIGHT):

        self.data_source = data_source
        self.performance_metrics = performance_metrics or {}
        self.max_word_length = max_word_length
        self.target_win_rate = target_win_rate / 100.0
        self.max_weight = max_weight
        self.weights = self.precompute_weights()

    def precompute_weights(self):
        # Find all unique word lengths in the dataset
        all_word_lengths = set(
            int(game_info['word_length']) for game_info in self.data_source)

        # Calculate raw weights for all word lengths
        raw_weights = {}
        for word_length in all_word_lengths:
            raw_weights[word_length] = self.calculate_weight(
                word_length,
                self.performance_metrics.get(
                    word_length, {}).get('win_rate', 50) / 100,
                self.performance_metrics.get(word_length, {}).get(
                    'average_attempts_used', 6)
            )

        # Normalize weights
        total_weight_sum = sum(raw_weights.values())
        normalized_weights = {
            word_length: weight / total_weight_sum for word_length,
            weight in raw_weights.items()}
        return normalized_weights

    def calculate_weight(self, word_length, win_rate, attempts):
        length_weight = (word_length / self.max_word_length) * 2
        win_rate_deviation = abs(self.target_win_rate - win_rate)
        win_rate_weight = 1 + (win_rate_deviation * 2)
        attempts_weight = 1 if attempts <= 6 else (10 - attempts) / 4
        total_weight = length_weight * win_rate_weight * attempts_weight
        return max(min(int(total_weight), self.max_weight), 1)

    def __iter__(self):
        n_samples = min(MAX_INDICES, len(self.data_source))

        # Calculating raw probabilities for each index
        raw_probabilities = [self.weights.get(int(self.data_source[idx]['word_length']), 0)
                             for idx in range(len(self.data_source))]

        # Normalizing the probabilities
        total_prob_sum = sum(raw_probabilities)
        normalized_probabilities = [
            prob / total_prob_sum for prob in raw_probabilities]

        # Check the sum of normalized probabilities
        if not np.isclose(sum(normalized_probabilities), 1):
            raise ValueError(
                "Sum of normalized probabilities does not equal 1")

        counts = np.random.multinomial(n_samples, normalized_probabilities)
        sampled_indices = np.where(counts > 0)[0]

        for index in sampled_indices:
            for _ in range(counts[index]):
                yield index

    def __len__(self):
        return min(MAX_INDICES, len(self.data_source))


# # dont change below

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


def stratified_sample_by_length(word_list, num_stratified_samples):
    word_groups = group_words_by_length(word_list)
    total_words = len(word_list)
    stratified_sampled_words = []

    for word_length, words in word_groups.items():
        proportion = len(words) / total_words
        num_samples_per_group = max(
            1, round(proportion * num_stratified_samples))

        # Randomly sample words within each length group
        sampled_words = random.sample(
            words, min(len(words), num_samples_per_group))
        stratified_sampled_words.extend(sampled_words)

    return stratified_sampled_words


def group_words_by_length_and_uniqueness(word_list):
    word_groups = {}
    for word in word_list:
        key = (len(word), len(set(word)))
        word_groups.setdefault(key, []).append(word)
    return word_groups


def stratified_sample_by_length_and_uniqueness(word_list, num_stratified_samples):
    word_groups = group_words_by_length_and_uniqueness(word_list)
    total_words = len(word_list)
    stratified_sampled_words = []

    for (word_length, unique_chars), words in word_groups.items():
        proportion = len(words) / total_words
        num_samples_per_group = max(
            1, round(proportion * num_stratified_samples))

        # Randomly sample words within each length and uniqueness group
        sampled_words = random.sample(
            words, min(len(words), num_samples_per_group))
        stratified_sampled_words.extend(sampled_words)

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
