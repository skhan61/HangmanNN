import os
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from scr.custom_sampler import *
from scr.custom_sampler import PerformanceBasedSampler
# Assuming this contains necessary dataset classes and functions
from scr.dataset import *  # Assuming necessary dataset classes and functions are here
from scr.utils import *


class HangmanDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset,
                 batch_size, collate_fn, use_custom_sampler=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.use_custom_sampler = use_custom_sampler

    def train_dataloader(self):
        # # Determine the sampler to use based on the epoch
        # if self.use_custom_sampler and hasattr(self.trainer.model, 'last_eval_metrics'):
        #     # print(f"From Performence Sampler...")
        #     # metrics = self.trainer.model.last_eval_metrics
        #     sampler = PerformanceBasedSampler(
        #         self.train_dataset, self.trainer.model.last_eval_metrics)
        # else:
        #     # print(f"From RandomSampler...")
        #     sampler = RandomSampler(self.train_dataset)

        # # Using BatchSampler
        # batch_sampler = BatchSampler(
        #     sampler, batch_size=self.batch_size, drop_last=False)

        # return DataLoader(self.train_dataset, batch_sampler=batch_sampler,
        #                   collate_fn=self.collate_fn, num_workers=os.cpu_count()
        #                   or 4, prefetch_factor=2,
        #                   pin_memory=True)

        return DataLoader(self.train_dataset, collate_fn=self.collate_fn,
                          num_workers=os.cpu_count()
                          or 4, prefetch_factor=2,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() or 1,
        )

    def use_performance_based_sampling(self, use_sampler: bool):
        self.use_custom_sampler = use_sampler
