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
    def __init__(self, train_dataset, val_dataset,
                 batch_size, collate_fn,
                 performance_metrics=None,
                 threshold_win_rate=50):

        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.performance_metrics = performance_metrics
        self.threshold_win_rate = threshold_win_rate

    def create_sampler(self, performance_metrics=None):
        if performance_metrics:
            return PerformanceBasedSampler(
                self.train_dataset, performance_metrics,
                self.threshold_win_rate)
        else:
            return RandomSampler(self.train_dataset)

    def train_dataloader(self):
        # Create a new dataloader every time this method is called
        # This ensures that the latest performance metrics are used
        sampler = self.create_sampler(self.performance_metrics)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 4,
            prefetch_factor=2,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 1
        )

    def update_train_dataloader(self, new_performance_metrics):
        # Update the performance metrics
        self.performance_metrics = new_performance_metrics
        # The train dataloader will be recreated with the updated metrics on the next call
