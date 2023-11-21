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


# def group_words_by_length(word_list):
#     word_groups_by_length = defaultdict(list)
#     for word in word_list:
#         word_groups_by_length[len(word)].append(word)
#     #     print(f"Added '{word}' to group of length {len(word)}")  # Debug line
#     # print(f"Grouped words by length: {word_groups_by_length}")  # Debug line
#     return word_groups_by_length


# def stratified_sample_by_length_and_frequency(word_list, word_freq_dict,
#                                               num_stratified_samples):
#     word_groups = group_words_by_length(word_list)
#     total_words = sum(len(group) for group in word_groups.values())
#     stratified_sampled_words = []

#     # print(f"Total words: {total_words}")  # Debug line

#     for word_length, words in word_groups.items():
#         # print(f"\nProcessing word length group: {word_length}")  # Debug line
#         words_sorted_by_freq = sorted(
#             words, key=lambda w: word_freq_dict.get(w, 0))
#         # print(f"Words sorted by frequency: {words_sorted_by_freq}")  # Debug line
#         half_len = len(words) // 2
#         high_freq_words = words_sorted_by_freq[:half_len]
#         low_freq_words = words_sorted_by_freq[half_len:]
#         # print(f"High frequency words: {high_freq_words}")  # Debug line
#         # print(f"Low frequency words: {low_freq_words}")  # Debug line
#         # num_samples_per_group = int(num_stratified_samples * len(words) / total_words)
#         num_samples_per_group = max(
#             1, int(num_stratified_samples * len(words) / total_words))
#         high_freq_samples = min(len(high_freq_words),
#                                 num_samples_per_group // 2)
#         low_freq_samples = min(len(low_freq_words), num_samples_per_group // 2)
#         # print(f"Sampling {high_freq_samples} from high frequency and {low_freq_samples} from low frequency")  # Debug line
#         if high_freq_words:
#             stratified_sampled_words.extend(
#                 random.sample(high_freq_words, high_freq_samples))
#         if low_freq_words:
#             stratified_sampled_words.extend(
#                 random.sample(low_freq_words, low_freq_samples))
#         # print(f"Sampled words so far: {stratified_sampled_words}")  # Debug line
#     return stratified_sampled_words
gc.collect()


# import random

# def sample_scenarios(scenarios, base_sample_size, \
#     max_samples_per_length=15, always_include_masked_state=None):
#     sampled = []
#     word_length_categories = set([len(s['word']) for s in scenarios])

#     for length in word_length_categories:
#         length_scenarios = [s for s in scenarios if len(s['word']) == length]
#         total_samples_for_length = 0

#         # Always include the fully masked state scenario if provided
#         if always_include_masked_state:
#             masked_state_scenarios = [s for s in length_scenarios \
#                 if s['initial_state'] == always_include_masked_state]

#             for scenario in masked_state_scenarios:
#                 sampled.append(scenario)
#                 total_samples_for_length += 1

#         # Continue with other categories
#         for category in ["easy_win", "easy_lose", "medium_win", "medium_lose", \
#             "hard_win", "hard_lose"]:
#             cat_scenarios = [s for s in length_scenarios if s['difficulty'] \
#                 == category.split('_')[0] and s['outcome'] == category.split('_')[1]]

#             available_samples = max_samples_per_length - total_samples_for_length
#             if available_samples <= 0:
#                 break

#             sample_size = min(len(cat_scenarios), base_sample_size, available_samples)
#             sampled.extend(random.sample(cat_scenarios, sample_size))
#             total_samples_for_length += sample_size

#     # # Debug: Check for inclusion of fully masked state scenarios in the final sample
#     # for scenario in sampled:
#     #     initial_state = scenario.get('initial_state')
#     #     if initial_state == always_include_masked_state:
#     #         print(f"Debug: Fully masked state scenario included for word '{scenario['word']}'")
#     #     elif initial_state is not None:
#     #         print(f"Debug: Other initial state scenario for word '{scenario['word']}'")
#     #     else:
#     #         print(f"Debug: No initial state provided for word '{scenario['word']}'")

#     return sampled
