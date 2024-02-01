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
        self.batch_sampler = None  # To be potentially used with PerformanceBasedSampler

    def train_dataloader(self):
        # Initialize batch_sampler based on the current performance metrics
        if self.performance_metrics:
            print('Performance batch sampler activated...')
            self.batch_sampler = PerformanceBasedSampler(
                self.train_dataset, self.performance_metrics, self.batch_size)
        else:
            print('Standard batching activated...')
            self.batch_sampler = None  # Reset to None to use default batching

        # Check if the dataset is empty
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        # Conditional DataLoader initialization based on batch_sampler presence
        if self.batch_sampler:
            return DataLoader(self.train_dataset,
                              batch_sampler=self.batch_sampler,  # Use batch_sampler
                              collate_fn=self.collate_fn,
                              num_workers=os.cpu_count() or 4, prefetch_factor=2,
                              pin_memory=True)
        else:
            return DataLoader(self.train_dataset,
                              batch_size=self.batch_size,  # Use fixed batch size
                              collate_fn=self.collate_fn,
                              shuffle=True,  # Now shuffling should work
                              num_workers=os.cpu_count() or 4, prefetch_factor=2,
                              pin_memory=True)

    def val_dataloader(self):
        # Check if the validation dataset is empty
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty.")

        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=os.cpu_count() or 1)

    def update_performance_metrics(self, new_performance_metrics):
        self.performance_metrics = new_performance_metrics

        # Update or initialize batch_sampler with new performance metrics
        if self.batch_sampler:
            self.batch_sampler.update_performance_metrics(
                new_performance_metrics)
        else:
            self.batch_sampler = PerformanceBasedSampler(
                self.train_dataset, new_performance_metrics, self.batch_size)
