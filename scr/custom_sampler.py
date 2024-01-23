import gc
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from scr.dataset import *

# class PerformanceBasedSampler(Sampler):
#     def __init__(self, dataset, performance_metrics, batch_size):
#         self.dataset = dataset
#         self.performance_metrics = performance_metrics
#         self.batch_size = batch_size
#         self.target_pairs = self._select_target_pairs()
#         self.last_used_pairs = None

#     def _select_target_pairs(self):
#         metrics = [(name, score) for name, score in self.performance_metrics.items(
#         ) if "avg_miss_penalty_" in name]
#         metrics.sort(key=lambda x: x[1], reverse=True)
#         target_pairs = [re.findall(r"'([^']*)'", name)
#                         for name, _ in metrics[:self.batch_size]]
#         return [tuple(pair) for pair in target_pairs if len(pair) == 2]

#     def __iter__(self):
#         if self.target_pairs != self.last_used_pairs:
#             self.last_used_pairs = self.target_pairs
#             self.dataset.rebuild_pair_index(self.target_pairs)

#         valid_indices = []
#         for difficulty, outcome in self.target_pairs:
#             if (difficulty, outcome) in self.dataset.pair_index:
#                 valid_indices.extend([(difficulty, outcome, file_idx, local_idx)
#                                       for file_idx, local_idx in self.dataset.pair_index[(difficulty, outcome)]])
#         return iter(valid_indices)

#     def __len__(self):
#         return sum(len(self.dataset.pair_index[pair]) for pair in self.target_pairs)

#     def update_target_pairs(self, new_performance_metrics):
#         self.performance_metrics = new_performance_metrics
#         new_target_pairs = self._select_target_pairs()
#         if new_target_pairs != self.last_used_pairs:
#             self.target_pairs = new_target_pairs
#             self.last_used_pairs = new_target_pairs
#             self.dataset.rebuild_pair_index(new_target_pairs)


class PerformanceBasedSampler(Sampler):
    def __init__(self, dataset, performance_metrics, batch_size, score_threshold=0):
        self.dataset = dataset
        self.performance_metrics = performance_metrics

        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.target_pairs = self._select_target_pairs()
        self.last_used_pairs = None

    def _select_target_pairs(self):
        # Initialize an empty list for target pairs
        target_pairs = []

        # print(self.performance_metrics.keys())

        # Iterate through performance metrics and apply the selection criteria
        for metric_name, metric_value in self.performance_metrics.items():
            if metric_name.startswith('win_rate_') and metric_value <= 50:
                # Extract word length from metric name
                word_length = int(metric_name.split('_')[-1])
                # Check the corresponding average attempts metric
                avg_attempts_metric_name = f'avg_attempts_{word_length}'
                avg_attempts = self.performance_metrics.get(
                    avg_attempts_metric_name, 0)

                # Apply the criteria for average attempts
                if avg_attempts >= 4:
                    target_pairs.append((word_length,))

        # Limit the number of target pairs based on batch size
        target_pairs = target_pairs[:self.batch_size]
        return target_pairs

    def __iter__(self):
        # Rebuild pair index if target pairs have changed
        if self.target_pairs != self.last_used_pairs:
            self.last_used_pairs = self.target_pairs
            self.dataset.rebuild_pair_index(self.target_pairs)

        valid_indices = []
        for pair in self.target_pairs:
            if len(pair) == 2:  # For (difficulty, outcome) pairs
                difficulty, outcome = pair
                if (difficulty, outcome) in self.dataset.pair_index:
                    valid_indices.extend([(difficulty, outcome, file_idx, local_idx)
                                          for file_idx, local_idx in self.dataset.pair_index[(difficulty, outcome)]])
            elif len(pair) == 1:  # For (word_length,) pairs
                word_length = pair[0]
                if word_length in self.dataset.word_length_index:
                    valid_indices.extend(
                        [(word_length,) for _ in self.dataset.word_length_index[word_length]])

        return iter(valid_indices)

    def __len__(self):
        length = 0
        for pair in self.target_pairs:
            if len(pair) == 2 and pair in self.dataset.pair_index:
                length += len(self.dataset.pair_index[pair])
            elif len(pair) == 1 and pair[0] in self.dataset.word_length_index:
                length += len(self.dataset.word_length_index[pair[0]])
        return length

        def update_target_pairs(self, new_performance_metrics):
            self.performance_metrics = new_performance_metrics
            new_target_pairs = self._select_target_pairs()
            if new_target_pairs != self.last_used_pairs:
                self.target_pairs = new_target_pairs
                self.last_used_pairs = new_target_pairs
                self.dataset.rebuild_pair_index(new_target_pairs)

# =================================================
# class PerformanceBasedSampler(Sampler):
#     def __init__(self, dataset, performance_metrics, batch_size, score_threshold=0):
#         self.dataset = dataset
#         self.performance_metrics = performance_metrics
#         self.batch_size = batch_size
#         self.score_threshold = score_threshold
#         self.target_pairs = self._select_target_pairs()
#         self.last_used_pairs = None

    # def _select_target_pairs(self):
    #     # Filter metrics that exceed the score threshold
    #     filtered_metrics = [
    #         (name, score) for name, score in self.performance_metrics.items()
    #         if "avg_miss_penalty_" in name and score >= -1e10
    #     ]  # self.score_threshold

    #     # Sort the filtered metrics in descending order of score
    #     filtered_metrics.sort(key=lambda x: x[1], reverse=True)

    #     # Select the top metrics as target pairs
    #     target_pairs = [re.findall(r"'([^']*)'", name)
    #                     for name, _ in filtered_metrics[:self.batch_size]]
    #     return [tuple(pair) for pair in target_pairs if len(pair) == 2]

#     def __iter__(self):
#         # Rebuild pair index if target pairs have changed
#         if self.target_pairs != self.last_used_pairs:
#             self.last_used_pairs = self.target_pairs
#             self.dataset.rebuild_pair_index(self.target_pairs)

#         valid_indices = []
#         for difficulty, outcome in self.target_pairs:
#             if (difficulty, outcome) in self.dataset.pair_index:
#                 valid_indices.extend([(difficulty, outcome, file_idx, local_idx)
#                                       for file_idx, local_idx in self.dataset.pair_index[(difficulty, outcome)]])
#         return iter(valid_indices)

#     def __len__(self):
#         return sum(len(self.dataset.pair_index[pair]) for pair in self.target_pairs)

#     def update_target_pairs(self, new_performance_metrics):
#         self.performance_metrics = new_performance_metrics
#         new_target_pairs = self._select_target_pairs()
#         if new_target_pairs != self.last_used_pairs:
#             self.target_pairs = new_target_pairs
#             self.last_used_pairs = new_target_pairs
#             self.dataset.rebuild_pair_index(new_target_pairs)


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
