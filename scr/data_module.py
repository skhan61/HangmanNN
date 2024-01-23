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


class HangmanDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.performance_metrics = None  # Initialize as None
        self.sampler = None

    def train_dataloader(self):
        # print("Performance Metrics:", data_module.performance_metrics)
        # Initialize sampler based on the current performance metrics
        if self.performance_metrics:
            print('Performance sampler kicked...')
            self.sampler = PerformanceBasedSampler(
                self.train_dataset, self.performance_metrics, self.batch_size)
        else:
            print('Normal sampler kicked...')
            self.sampler = None  # Or some default sampler

        # Check if the dataset is empty
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          sampler=self.sampler,
                          collate_fn=self.collate_fn,
                          num_workers=os.cpu_count() or 4, prefetch_factor=2,
                          pin_memory=True)

    def val_dataloader(self):
        # Check if the validation dataset is empty
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty.")

        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=os.cpu_count() or 1)

    def update_performance_metrics(self, new_performance_metrics):
        self.performance_metrics = new_performance_metrics

        # Update sampler with new performance metrics
        if self.sampler:
            self.sampler.update_target_pairs(new_performance_metrics)
        else:
            self.sampler = PerformanceBasedSampler(
                self.train_dataset, new_performance_metrics, self.batch_size)

        # Note: The DataLoader will be automatically updated in the next call to train_dataloader()


# class HangmanDataModule(LightningDataModule):
#     def __init__(self, train_dataset, val_dataset, batch_size, collate_fn):
#         super().__init__()
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.batch_size = batch_size
#         self.collate_fn = collate_fn
#         self.performance_metrics = None  # Initialize as None

#     def train_dataloader(self):
#         # Initialize sampler based on the current performance metrics
#         if self.performance_metrics:
#             print('Performence sampler kicked...')
#             sampler = PerformanceBasedSampler(
#                 self.train_dataset, self.performance_metrics, self.batch_size)
#         else:
#             sampler = None

#         # Check if the dataset is empty
#         if len(self.train_dataset) == 0:
#             raise ValueError("Training dataset is empty.")

#         # sampler = RandomSampler(self.train_dataset)

#         return DataLoader(self.train_dataset, batch_size=self.batch_size,
#                           sampler=sampler,
#                           collate_fn=self.collate_fn,
#                           num_workers=os.cpu_count() or 4, prefetch_factor=2,
#                           pin_memory=True)

#     def val_dataloader(self):
#         # Check if the validation dataset is empty
#         if len(self.val_dataset) == 0:
#             raise ValueError("Validation dataset is empty.")

#         return DataLoader(self.val_dataset, batch_size=self.batch_size,
#                           collate_fn=self.collate_fn, num_workers=os.cpu_count() or 1)

#     def update_performance_metrics(self, new_metrics):
#         # Update performance metrics
#         self.performance_metrics = new_metrics

#         # The actual reinitialization of the dataloader will be handled
#         # by the Trainer's `reload_dataloaders_every_n_epochs` setting.

#     def update_performance_metrics(self, new_performance_metrics):
#         # Update the performance metrics in the sampler
#         if hasattr(self, 'sampler'):
#             self.sampler.update_target_pairs(new_performance_metrics)

#         # Recreate the DataLoader with the updated sampler
#         self.train_dataloader = DataLoader(
#             self.train_dataset, batch_sampler=self.sampler)
