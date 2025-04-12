import os
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io
import torch
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


class MatToPklConverter:
    """Конвертирует .mat файлы в train/test .pkl"""

    @staticmethod
    def convert_folder(
        input_dir: str, output_train: str, output_test: str, test_size: float = 0.2
    ):
        """
        Args:
            input_dir: Папка с .mat файлами
            output_train: Путь для сохранения train.pkl
            output_test: Путь для сохранения test.pkl
            test_size: Доля тестовых данных
        """
        input_dir = Path(input_dir)
        records = []

        # Чтение .mat файлов
        for mat_file in input_dir.glob("*.mat"):
            data = scipy.io.loadmat(str(mat_file))
            emg = data["emg"]
            stimulus = data["stimulus"].flatten()

            for i in range(len(stimulus)):
                if stimulus[i] != 0:  # Пропускаем покой
                    records.append({"emg": emg[i], "stimulus": int(stimulus[i])})

        # Разделение на train/test
        df = pd.DataFrame(records)
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df["stimulus"]
        )

        # Сохранение
        train_df.to_pickle(output_train)
        test_df.to_pickle(output_test)
        print(f"Сохранено: {output_train} ({len(train_df)} записей)")
        print(f"Сохранено: {output_test} ({len(test_df)} записей)")


class Nina1Dataset(torch.utils.data.Dataset):
    """Кастомный класс, используемый для формирования датасета из данных для нашей задачи"""

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame с колонками 'emg' и 'stimulus'
        """
        self.dataframe = data

        # Проверка структуры данных
        if not all(col in data.columns for col in ["emg", "stimulus"]):
            raise ValueError("DataFrame must contain 'emg' and 'stimulus' columns")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        emg = row["emg"]
        stimulus = row["stimulus"]

        # Обработка EMG сигнала
        if isinstance(emg, list) and len(emg) == 1:
            emg = emg[0]

        # Дополнение нулями до 500 отсчетов
        data = emg[:500]
        if len(data) < 500:
            data = np.concatenate((data, np.zeros((500 - len(data), 10))), axis=0)

        # Изменение формы (25, 20, 10)
        input_data = data.reshape((25, 20, 10))

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(stimulus, dtype=torch.long),
        )


class MyDataModule(pl.LightningDataModule):
    """В этом файле ведется вся работы с данными. Для этого есть специальный встроенный модуль.
    Модуль DataModule стандартизирует разбиение на train, test, val, подготовку данных и
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

    """DataModule для работы с готовыми .pkl файлами"""

    def __init__(
        self,
        data_root: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_size: float = 0.2,
    ):
        """
        Args:
            data_root: Корневая папка с данными
            batch_size: Размер батча
            num_workers: Число workers для DataLoader
            val_size: Доля валидационных данных от train
        """
        super().__init__()
        self.data_root = Path(data_root).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size

        # Пути к файлам
        self.train_pkl = self.data_root / "train_data" / "ninaprodb1train.pkl"
        self.test_pkl = self.data_root / "test_data" / "ninaprodb1test.pkl"

        # Проверка существования файлов
        if not self.train_pkl.exists():
            raise FileNotFoundError(f"Train file not found: {self.train_pkl}")
        if not self.test_pkl.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_pkl}")

    def setup(self, stage: Optional[str] = None):
        """Загрузка данных и разделение на train/val/test"""
        # Загрузка train и разделение на train/val
        train_df = pd.read_pickle(self.train_pkl)
        train_data, val_data = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=42,
            stratify=train_df["stimulus"],
        )

        # Загрузка test
        test_df = pd.read_pickle(self.test_pkl)

        # Создание датасетов
        self.train_ds = Nina1Dataset(train_data)
        self.val_ds = Nina1Dataset(val_data)
        self.test_ds = Nina1Dataset(test_df)

        print(f"Train samples: {len(self.train_ds)}")
        print(f"Val samples: {len(self.val_ds)}")
        print(f"Test samples: {len(self.test_ds)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


if __name__ == "__main__":
    # Абсолютный путь до корня проекта
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"

    train_path = DATA_ROOT / "train_data" / "ninaprodb1train.pkl"
    test_path = DATA_ROOT / "test_data" / "ninaprodb1test.pkl"

    print("Текущая рабочая директория:", os.getcwd())
    print(f"Train path: {train_path} | Существует: {train_path.exists()}")
    print(f"Test path: {test_path} | Существует: {test_path.exists()}")
    print(f"Содержимое data/: {list(DATA_ROOT.glob('*/*'))}")

    if not train_path.exists() or not test_path.exists():
        raise RuntimeError("Файлы данных не найдены! Проверьте структуру папок")

    # Передаём уже готовый абсолютный Path
    dm = MyDataModule(data_root=DATA_ROOT, batch_size=32, num_workers=4, val_size=0.2)

    dm.setup()

    # Проверка загрузки
    batch = next(iter(dm.train_dataloader()))
    print("Пример батча:")
    print("EMG data shape:", batch[0].shape)  # [32, 25, 20, 10]
    print("Labels shape:", batch[1].shape)  # [32]
