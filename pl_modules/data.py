from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split


# Кастомный класс Dataset для PyTorch
class Nina1Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.dataframe = pd.read_pickle(path)

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
    """Модуль DataModule стандартизирует разбиение на train, test, val, подготовку данных и
    их преобразование. Основным преимуществом является согласованное разбиение данных,
    подготовка данных и преобразования в разных моделях.

    Пример::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # скачивание, разбиение, и т.д....
                # вызывается только на 1 GPU/TPU при распределении
            def setup(self, stage):
                # разбиение и доп.операции с выборками (val/train/test)
                # вызывается на каждом процессе DDP
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
                # отчистка после валидации или тестирования
                # вызывается на каждом процессе DDP
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Используйте это для загрузки и подготовки данных. Загрузка и сохранение
        данных в нескольких процессах (с распределенными настройками) приведет к
        повреждению данных.
        Lightning гарантирует, что этот метод вызывается только в рамках одного процесса,
        поэтому вы можете безопасно добавить свою логику загрузки в него.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Вызывается в начале процесса fit (train + validate), проверки, тестирования
        или прогнозирования.
        Это хороший инструмент, когда вам нужно динамически создавать модели или что-то в них
        корректировать. Этот инструмент вызывается в каждом процессе при использовании DDP.

        setup вызывается из каждого процесса на всех узлах. Здесь рекомендуется задать состояние.
        """
        full_dataset = Nina1Dataset(self.train_path)
        self.train_dataset, self.val_dataset = train_test_split(
            full_dataset, test_size=0.3, random_state=21
        )
        self.test_dataset = Nina1Dataset(self.test_path)

    def teardown(self, stage: str) -> None:
        """Вызывается в конце обучения (train + validate), валидации, тестирования или предсказания.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = MyDataModule()
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("Batch shape:", batch[0].shape)
