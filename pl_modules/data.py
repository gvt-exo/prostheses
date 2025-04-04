from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# Кастомный класс Dataset для PyTorch
class Nina1DatasetPyTorch(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        emg_data = row["emg"]

        # Обработка данных как в оригинальном классе
        if isinstance(emg_data, list) and len(emg_data) == 1:
            emg_data = emg_data[0]

        data = emg_data[:500]
        if len(data) < 500:
            data = np.concatenate((data, np.zeros((500 - len(data), 10))), axis=0)

        # Изменение формы данных (25, 20, 10)
        input_data = data.reshape((25, 20, 10))
        label = row["stimulus"]

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class MyDataModule(pl.LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and
    transforms. The main advantage is consistent data splits, data preparation and
    transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

    def prepare_data(self):
        """Use this to download and prepare data. Downloading and saving data with
        multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures this method is called only within a single process, so you can
        safely add your downloading logic within.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something
        about them. This hook is called on every process when using DDP.

        setup is called from every process across all the nodes. Setting state here is
        recommended.
        """
