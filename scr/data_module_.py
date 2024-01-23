import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler,
                              WeightedRandomSampler)
from tqdm import tqdm

from scr.custom_sampler import *
from scr.custom_sampler import PerformanceBasedSampler
# Assuming this contains necessary dataset classes and functions
from scr.dataset import *  # Assuming necessary dataset classes and functions are here
from scr.utils import *


def calculate_weights(dataset, win_weight_multiplier=2.0):
    # Calculate class counts for each (word_length, difficulty, \
    # outcome) combination
    class_counts = {key: len(indices)
                    for key, indices in dataset.pair_index.items()}

    # Calculate word length frequencies
    word_length_counts = defaultdict(int)
    for key in class_counts:
        word_length = key[0]  # The first element of the key is the word length
        word_length_counts[word_length] += class_counts[key]

    sample_weights = []

    # Iterate over each group and its indices in the pair_index
    for group_key, indices in dataset.pair_index.items():
        word_length = group_key[0]
        outcome = group_key[2]  # The third element of the key is the outcome
        group_weight_factor = 1.0 / len(indices)
        word_length_weight_factor = 1.0 / word_length_counts[word_length]

        # Combine group weight factor and word length weight factor
        combined_weight = group_weight_factor * word_length_weight_factor

        # Increase weight for 'win' outcome
        if outcome == 'win':
            combined_weight *= win_weight_multiplier

        # Apply the combined weight to each sample in the group
        group_weights = [combined_weight] * len(indices)
        sample_weights.extend(group_weights)

    return torch.DoubleTensor(sample_weights)


class HangmanDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        # Pre-calculate weights for each sample in the training dataset
        self.sample_weights = calculate_weights(self.train_dataset)

    def train_dataloader(self):
        # Use pre-calculated weights for the weighted random sampler
        weighted_sampler = WeightedRandomSampler(
            self.sample_weights, len(self.sample_weights))

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=weighted_sampler,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 4,
            prefetch_factor=2,
            pin_memory=True
        )

    def val_dataloader(self):
        # For the validation dataset, use a standard non-weighted sampler
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 1
        )


# class HangmanDataModule(LightningDataModule):
#     def __init__(self, train_dataset, val_dataset, batch_size, collate_fn):
#         super().__init__()
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.batch_size = batch_size
#         self.collate_fn = collate_fn
#         self.sample_weights = None

#     def calculate_sample_weights(self):
#         # Calculate weights only if they haven't been calculated yet
#         if self.sample_weights is None:
#             self.sample_weights = calculate_weights(self.train_dataset)
#         return self.sample_weights

#     def train_dataloader(self):
#         # Calculate weights for each sample in the training dataset
#         sample_weights = self.calculate_sample_weights()

#         # Create a weighted random sampler for the training dataset
#         weighted_sampler = WeightedRandomSampler(
#             sample_weights, len(sample_weights))

#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             sampler=weighted_sampler,
#             collate_fn=self.collate_fn,
#             num_workers=os.cpu_count() or 4,
#             prefetch_factor=2,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         # For the validation dataset, we use a standard non-weighted sampler
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.collate_fn,
#             num_workers=os.cpu_count() or 1
#         )


# class HangmanDataModule(LightningDataModule):
#     def __init__(self, train_dataset,
#                  val_dataset, batch_size,
#                  collate_fn, performance_metrics=None,
#                  threshold_win_rate=50):

#         super().__init__()
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.batch_size = batch_size
#         self.collate_fn = collate_fn
#         self.performance_metrics = performance_metrics
#         self.threshold_win_rate = threshold_win_rate
#         self.sampler = None  # Initialize the sampler attribute

#     def create_sampler(self, performance_metrics=None):
#         if performance_metrics:
#             # Ensure performance_metrics is in the format of items() from a dictionary
#             performance_metrics_items = performance_metrics.items() if isinstance(
#                 performance_metrics, dict) else performance_metrics
#             self.sampler = PerformanceBasedSampler(
#                 self.train_dataset, performance_metrics_items, self.threshold_win_rate)
#         else:
#             self.sampler = RandomSampler(self.train_dataset)
#         return self.sampler

#     def train_dataloader(self):
#         sampler = self.create_sampler(self.performance_metrics)
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             sampler=sampler,
#             collate_fn=self.collate_fn,
#             num_workers=os.cpu_count() or 4,
#             prefetch_factor=2,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.collate_fn,
#             num_workers=os.cpu_count() or 1
#         )

#     def update_performance_metrics(self, new_metrics):
#         # Ensure new_metrics is in the format of items() from a dictionary
#         new_metrics_items = new_metrics.items() if isinstance(
#             new_metrics, dict) else new_metrics
#         self.performance_metrics = new_metrics_items
#         if self.sampler and hasattr(self.sampler, 'update_target_pairs'):
#             self.sampler.update_target_pairs(new_metrics_items)


# class HangmanDataModule(LightningDataModule):
#     def __init__(self, train_dataset, val_dataset,
#                  batch_size, collate_fn, performance_metrics=None,
#                  threshold_win_rate=50):
#         super().__init__()
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.batch_size = batch_size
#         self.collate_fn = collate_fn
#         self.performance_metrics = performance_metrics
#         self.threshold_win_rate = threshold_win_rate
#         self.sampler = None  # Initialize the sampler attribute

#     def create_sampler(self, performance_metrics=None):
#         if performance_metrics:
#             self.sampler = PerformanceBasedSampler(
#                 self.train_dataset, performance_metrics, self.threshold_win_rate
#             )
#         else:
#             self.sampler = RandomSampler(self.train_dataset)
#         return self.sampler

#     def train_dataloader(self):
#         sampler = self.create_sampler(self.performance_metrics)
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             sampler=sampler,
#             collate_fn=self.collate_fn,
#             num_workers=os.cpu_count() or 4,
#             prefetch_factor=2,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.collate_fn,
#             num_workers=os.cpu_count() or 1
#         )

#     def update_performance_metrics(self, new_metrics):
#         self.performance_metrics = new_metrics.items()
#         if self.sampler and hasattr(self.sampler, 'update_target_pairs'):
#             self.sampler.update_target_pairs(new_metrics)
