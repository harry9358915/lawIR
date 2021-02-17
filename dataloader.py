from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

#from BertMaxPdataset import BertMaxPDataset
from longformerdataset import longformerMaxpDataset
class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: longformerMaxpDataset,
        batch_size: int,
        shuffle: str = False,
        num_workers: int = 0
    ) -> None:
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = dataset.collate
        )