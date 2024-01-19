import gc
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Sampler

from scr.dataset import *


class WordLengthSampler(Sampler):
    def __init__(self, dataset, target_word_lengths, num_samples=None):
        self.dataset = dataset
        self.target_word_lengths = target_word_lengths
        self.num_samples = num_samples or sum(len(dataset.word_length_index[wl])
                                              for wl in target_word_lengths)

    def __iter__(self):
        for _ in range(self.num_samples):
            # Randomly select a word length from the target lengths
            word_length = random.choice(self.target_word_lengths)
            # Randomly select a game record index for that word length
            idx = random.choice(
                range(len(self.dataset.word_length_index[word_length])))
            yield word_length, idx

# # Example: create a sampler that targets word lengths 5 and 10
# target_word_lengths = [5, 10]

# sampler = WordLengthSampler(dataset, target_word_lengths)


class PerformanceBasedSampler(Sampler):
    def __init__(self, dataset, performance_metrics,
                 threshold_win_rate, num_samples=None):
        self.dataset = dataset
        self.performance_metrics = performance_metrics
        self.threshold_win_rate = threshold_win_rate
        self.target_word_lengths = self._select_target_lengths()
        self.num_samples = num_samples or sum(len(self.dataset.word_length_index[wl])
                                              for wl in self.target_word_lengths)

    def _select_target_lengths(self):
        # Select word lengths with win rates below the threshold
        return [length for length, metrics in self.performance_metrics.items()
                if metrics['win_rate'] < self.threshold_win_rate]

    def __iter__(self):
        sample_indices = []
        for word_length in self.target_word_lengths:
            indices = [(word_length, idx) for idx in
                       range(len(self.dataset.word_length_index[word_length]))]
            sample_indices.extend(indices)

        # Shuffle the sample indices to ensure diversity
        random.shuffle(sample_indices)

        return iter(sample_indices[:self.num_samples])

    def __len__(self):
        # Return the total number of samples to be drawn by this sampler
        return min(self.num_samples, sum(len(self.dataset.word_length_index[wl])
                                         for wl in self.target_word_lengths))

    def update_target_word_lengths(self, new_performance_metrics):
        self.performance_metrics = new_performance_metrics
        self.target_word_lengths = self._select_target_lengths()

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
