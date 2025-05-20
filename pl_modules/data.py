"""
Модуль для работы с данными ЭМГ.

Этот модуль содержит классы и функции для загрузки,
предобработки и аугментации данных электромиографии.
Включает в себя датасеты для обучения и валидации,
а также методы для работы с временными окнами.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


class Nina1Dataset(Dataset):
    """Кастомный класс, используемый для формирования датасета из данных для нашей задачи."""

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 500,  # Размер окна для сегментации сигнала
        sliding_window_size: int = 25,  # Размер скользящего окна
        step_size: int = 1,  # Шаг для скользящего окна
        is_train: bool = True,
    ):
        """Инициализация датасета.

        Args:
            data: DataFrame с колонками 'emg' и 'stimulus'
            window_size: Размер окна для сегментации сигнала (количество сэмплов)
            sliding_window_size: Размер скользящего окна для LSTM
            step_size: Шаг для скользящего окна (количество сэмплов)
            is_train: Флаг, указывающий, является ли датасет тренировочным
        """
        self.dataframe = data
        # Параметры для гибкой настройки размеров окон
        self.window_size = window_size
        self.sliding_window_size = sliding_window_size
        self.step_size = step_size
        self.is_train = is_train

        # Проверка структуры данных
        if not all(col in data.columns for col in ["emg", "stimulus"]):
            raise ValueError("DataFrame must contain 'emg' and 'stimulus' columns")

        # Предварительная обработка данных
        # Создаем списки для хранения обработанных окон и меток
        self.processed_data = []
        self.processed_labels = []

        # Обрабатываем каждый пример в датасете
        for idx in range(len(data)):
            row = data.iloc[idx]
            emg = row["emg"]
            stimulus = row["stimulus"]

            # Обработка случая, когда emg - список
            if isinstance(emg, list) and len(emg) == 1:
                emg = emg[0]

            # Обрезаем или дополняем сигнал до window_size
            if len(emg) > window_size:
                emg = emg[:window_size]  # Обрезаем лишние сэмплы
            elif len(emg) < window_size:
                # Дополняем нулями до нужного размера
                emg = np.pad(
                    emg, ((0, window_size - len(emg)), (0, 0)), mode="constant"
                )

            # Создаем скользящие окна с заданным шагом
            # Для каждого окна создаем отдельный пример
            for i in range(0, window_size - sliding_window_size + 1, step_size):
                window = emg[i : i + sliding_window_size]  # Вырезаем окно
                self.processed_data.append(window)
                self.processed_labels.append(stimulus)

    def __len__(self) -> int:
        """Возвращает количество примеров в датасете."""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает пример из датасета по индексу.

        Args:
            idx: Индекс примера

        Returns:
            Кортеж (данные, метка):
            - данные: тензор формы [sliding_window_size, num_channels]
            - метка: тензор с меткой класса
        """
        data = self.processed_data[idx]
        stimulus = self.processed_labels[idx]

        # Аугментация данных (только для тренировочного сета)
        if self.is_train:
            # Базовая нормализация перед аугментацией
            data = (data - np.mean(data, axis=0, keepdims=True)) / (
                np.std(data, axis=0, keepdims=True) + 1e-8
            )

            # 1. Масштабирование амплитуды (80% шанс)
            if np.random.random() < 0.8:
                scale_factor = np.random.uniform(0.7, 1.3, size=data.shape[1])
                data = data * scale_factor

            # 2. Добавление шума (70% шанс)
            if np.random.random() < 0.7:
                noise_factor = np.random.uniform(0, 0.1)
                data = data + np.random.normal(0, noise_factor, data.shape)

            # 3. Временной сдвиг (60% шанс)
            if np.random.random() < 0.6:
                # Уменьшенный диапазон сдвига для меньших окон
                shift = np.random.randint(-5, 5)
                data = np.roll(data, shift, axis=0)
                if shift > 0:
                    data[:shift, :] = 0
                elif shift < 0:
                    data[shift:, :] = 0

            # 4. Случайное обнуление каналов (30% шанс)
            if np.random.random() < 0.3:
                num_channels = np.random.randint(1, 3)
                channels = np.random.choice(data.shape[1], num_channels, replace=False)
                data[:, channels] = 0

            # 5. Случайное зеркальное отражение (20% шанс)
            if np.random.random() < 0.2:
                data = data[::-1].copy()

            # 6. Случайное изменение частоты дискретизации (50% шанс)
            if np.random.random() < 0.5:
                stretch_factor = np.random.uniform(0.9, 1.1)
                new_length = int(data.shape[0] * stretch_factor)
                x_old = np.linspace(0, 1, data.shape[0])
                x_new = np.linspace(0, 1, new_length)
                new_data = np.zeros((new_length, data.shape[1]))
                for i in range(data.shape[1]):
                    new_data[:, i] = np.interp(x_new, x_old, data[:, i])
                data = new_data
                # Приводим к размеру sliding_window_size
                if new_length > self.sliding_window_size:
                    data = data[: self.sliding_window_size, :]  # Обрезаем
                elif new_length < self.sliding_window_size:
                    # Дополняем нулями
                    data = np.pad(
                        data,
                        ((0, self.sliding_window_size - new_length), (0, 0)),
                        mode="constant",
                    )

            # Повторная нормализация после аугментации
            data = (data - np.mean(data, axis=0, keepdims=True)) / (
                np.std(data, axis=0, keepdims=True) + 1e-8
            )
        else:
            # Для валидации и теста только нормализация
            data = (data - np.mean(data, axis=0, keepdims=True)) / (
                np.std(data, axis=0, keepdims=True) + 1e-8
            )

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(stimulus, dtype=torch.long),
        )


class MyDataModule(pl.LightningDataModule):
    """DataModule для работы с готовыми .pkl файлами."""

    def __init__(
        self,
        data_root: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_size: float = 0.2,
        window_size: int = 500,  # Размер окна для сегментации
        sliding_window_size: int = 25,  # Размер скользящего окна
        step_size: int = 1,  # Шаг для скользящего окна
    ):
        """Инициализация модуля данных.

        Args:
            data_root: Корневая папка с данными
            batch_size: Размер батча
            num_workers: Число workers для DataLoader
            val_size: Доля валидационных данных от train
            window_size: Размер окна для сегментации сигнала
            sliding_window_size: Размер скользящего окна для LSTM
            step_size: Шаг для скользящего окна
        """
        super().__init__()
        self.data_root = Path(data_root).resolve()
        # Параметры для загрузки данных
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        # Параметры для гибкой настройки размеров окон
        self.window_size = window_size
        self.sliding_window_size = sliding_window_size
        self.step_size = step_size

        # Пути к файлам
        self.train_pkl = self.data_root / "train_data" / "ninaprodb1train.pkl"
        self.test_pkl = self.data_root / "test_data" / "ninaprodb1test.pkl"

        # Проверка существования файлов
        if not self.train_pkl.exists():
            raise FileNotFoundError(f"Train file not found: {self.train_pkl}")
        if not self.test_pkl.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_pkl}")

        print(f"Train path: {self.train_pkl}")
        print(f"Test path: {self.test_pkl}")

    def setup(self, stage: Optional[str] = None):
        """Загрузка данных и разделение на train/val/test."""
        # Загрузка train и разделение на train/val
        train_df = pd.read_pickle(self.train_pkl)

        # Анализ уникальных меток
        unique_classes = sorted(train_df["stimulus"].unique())
        print(f"Уникальные классы в данных: {unique_classes}")
        print(f"Количество уникальных классов: {len(unique_classes)}")

        # Оставляем только первые 17 классов для соответствия архиву Тани
        selected_classes = unique_classes[:17]
        train_df = train_df[train_df["stimulus"].isin(selected_classes)]
        print(f"Отобрано классов: {len(selected_classes)} -> {selected_classes}")

        # Используем только эти классы далее
        unique_classes = selected_classes

        # Создаем маппинг для преобразования меток
        class_mapping = {
            old_label: idx
            for idx, old_label in enumerate(sorted(set(train_df["stimulus"].unique())))
        }
        print(f"Маппинг классов: {class_mapping}")

        # Применяем маппинг к меткам
        train_df["stimulus"] = train_df["stimulus"].map(class_mapping)

        train_data, val_data = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=42,
            stratify=train_df["stimulus"],
        )

        # Загрузка test и применение того же маппинга
        test_df = pd.read_pickle(self.test_pkl)
        test_df["stimulus"] = test_df["stimulus"].map(class_mapping)

        # Проверяем, что все метки в правильном диапазоне
        all_labels = set(train_df["stimulus"].unique()) | set(
            test_df["stimulus"].unique()
        )
        print(f"Все метки после маппинга: {sorted(all_labels)}")
        assert max(all_labels) < len(
            unique_classes
        ), "Метки классов вне допустимого диапазона"

        # Создание датасетов с новыми параметрами
        # Для каждого набора данных используем свои параметры
        self.train_ds = Nina1Dataset(
            train_data,
            window_size=self.window_size,
            sliding_window_size=self.sliding_window_size,
            step_size=self.step_size,
            is_train=True,  # Включаем аугментацию для train
        )
        self.val_ds = Nina1Dataset(
            val_data,
            window_size=self.window_size,
            sliding_window_size=self.sliding_window_size,
            step_size=self.step_size,
            is_train=False,  # Отключаем аугментацию для val
        )
        self.test_ds = Nina1Dataset(
            test_df,
            window_size=self.window_size,
            sliding_window_size=self.sliding_window_size,
            step_size=self.step_size,
            is_train=False,  # Отключаем аугментацию для test
        )

        print(f"Train samples: {len(self.train_ds)}")
        print(f"Val samples: {len(self.val_ds)}")
        print(f"Test samples: {len(self.test_ds)}")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Возвращает загрузчик данных для обучения."""
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Возвращает загрузчик данных для валидации."""
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Возвращает загрузчик данных для тестирования."""
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
    dm = MyDataModule(data_root=DATA_ROOT, batch_size=32, num_workers=4)

    dm.setup()

    # Проверка загрузки
    batch = next(iter(dm.train_dataloader()))
    print("Пример батча:")
    print("EMG data shape:", batch[0].shape)  # [32, 25, 20, 10]
    print("Labels shape:", batch[1].shape)  # [32]
