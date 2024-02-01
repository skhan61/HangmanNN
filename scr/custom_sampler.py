import gc
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from torch.utils.data.sampler import Sampler

from scr.dataset import *
from scr.utils import *


class PerformanceBasedSampler(Sampler):
    def __init__(self, dataset, performance_metrics, batch_size, score_threshold=0.0001):
        self.dataset = dataset
        self.performance_metrics = performance_metrics
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        # Set total_samples before calling _calculate_sampling_probabilities
        self.total_samples = len(self.dataset)
        self.probabilities = self._calculate_sampling_probabilities()
        # Additional attributes to support incomplete last batch
        self.num_batches = np.ceil(
            self.total_samples / self.batch_size).astype(int)

    # Rest of the class implementation...

    def __iter__(self):
        all_indices = np.arange(self.total_samples)
        sampled_indices = np.random.choice(
            a=all_indices,
            size=self.total_samples,  # Change here to sample indices for all data
            replace=False,
            p=self.probabilities
        )
        start_idx = 0
        for batch_num in range(self.num_batches):
            end_idx = start_idx + self.batch_size
            # Handle last batch which may be incomplete
            batch_indices = sampled_indices[start_idx:end_idx]
            # Convert flat indices to (file_index, data_index) tuples
            sampled_tuples = [self.dataset.flat_index_to_tuple(
                idx) for idx in batch_indices]
            yield sampled_tuples
            # print(sampled_tuples)
            start_idx = end_idx

    def _calculate_sampling_probabilities(self):
        probabilities = np.full(self.total_samples, fill_value=1e-5)
        for idx, penalty in self.performance_metrics.items():
            if penalty >= self.score_threshold:
                probabilities[idx - 1] = penalty  # Assuming idx starts from 1
            else:
                probabilities[idx - 1] = 1e-5
        probabilities /= probabilities.sum()
        return probabilities

    def __len__(self):
        # Reflects the actual number of batches, including a possibly incomplete last batch
        return self.num_batches

    def update_performance_metrics(self, new_performance_metrics):
        self.performance_metrics = new_performance_metrics
        self.probabilities = self._calculate_sampling_probabilities()

        self.dataset.rebuild_seq_len_index() ## TODO: do we need it current implementation?

# ==========================================
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
