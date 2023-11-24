import gc
import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler

MAX_EXTRA_WEIGHT = 100  # Example value, adjust as needed
TARGET_WIN_RATE = 0.5   # This is an example value for scaling the weight
MAX_INDICES = 10000     # Example value, adjust as needed


class PerformanceBasedSampler(Sampler):
    def __init__(self, data_source, performance_metrics,
                 target_win_rate=0.5, max_weight=100, max_word_length=10):
        self.data_source = data_source
        self.performance_metrics = performance_metrics
        self.target_win_rate = target_win_rate
        self.max_weight = max_weight
        self.max_word_length = max_word_length  # Add this line
        self.indices = self.generate_indices()

    def generate_indices(self):
        indices = []
        for idx, (_, _, additional_info) in enumerate(self.data_source):
            word_length = len(additional_info['word'])
            difficulty = additional_info['difficulty']
            outcome = additional_info['outcome']

            # Access performance metrics
            performance = self.performance_metrics.get(
                word_length, {}).get(difficulty, {}).get(outcome, {})
            win_rate = performance.get('win_rate', 1)
            max_attempts = performance.get('max_attempts', 10)

            # # Debug prints
            # print(
            #     f"Word Length: {word_length}, Difficulty: {difficulty}, Outcome: {outcome}")
            # print(f"Performance Metrics: {performance}")
            # print(f"Win Rate: {win_rate}, Max Attempts: {max_attempts}")

            # Calculate weight
            # Adjust the weight calculation
            weight = self.calculate_weight(
                word_length, difficulty, outcome, win_rate, max_attempts)

            indices.extend([idx] * weight)

        random.shuffle(indices)
        return indices[:MAX_INDICES]

    def __iter__(self):
        # Returns an iterator over the generated indices
        return iter(self.indices)

    def __len__(self):
        # Returns the length of the generated indices
        return len(self.indices)

    def calculate_weight(self, word_length, difficulty, outcome, win_rate, max_attempts):
        # Factor in word length
        # Assuming max_word_length is defined
        length_weight = word_length / self.max_word_length

        # Differentiate between win and lose outcomes
        if outcome == 'win':
            # Higher weight for less frequent wins
            outcome_weight = (1 - win_rate) * 2
        else:  # 'lose'
            outcome_weight = win_rate * 2  # Higher weight for more frequent losses

        # Combine factors for total weight
        total_weight = length_weight * outcome_weight * (max_attempts / 10)
        return min(int(total_weight), self.max_weight)


# Example usage
# performance_metrics = { ... }
# sampler = PerformanceBasedSampler(dataset, performance_metrics, target_win_rate=0.6, max_weight=120)


# dont change below

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
