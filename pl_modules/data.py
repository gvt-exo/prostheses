import os
from glob import glob
from pathlib import Path
from typing import Optional
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import Dataset
from collections import deque

def load_mat_files_from_folder(folder_path, segment_length=500):
    folder_path = Path(folder_path)
    mat_files = list(folder_path.glob("*.mat"))
    records = []

    for file_path in mat_files:
        mat_data = scipy.io.loadmat(str(file_path))
        emg_data = mat_data.get("emg")  # (N, 10)
        stimulus = mat_data.get("stimulus").flatten()  # (N,)

        # Разбиваем на сегменты по segment_length
        num_segments = len(emg_data) // segment_length
        remaining_samples = len(emg_data) % segment_length

        # Работа с полными сегментами
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment_emg = emg_data[start:end]  # (segment_length, 10)
            segment_stimulus = stimulus[start:end]
            dominant_stimulus = np.argmax(np.bincount(segment_stimulus))
            
            records.append({
                "emg": segment_emg,
                "stimulus": int(dominant_stimulus)
            })

        # Последний неполный сегмент 
        if remaining_samples > 0:
            last_segment = emg_data[-remaining_samples:]  # (remaining_samples, 10)
            padded_segment = np.pad(
                last_segment,
                ((0, segment_length - remaining_samples), (0, 0)),
                mode='constant'
            )
            last_stimulus = stimulus[-remaining_samples:]
            dominant_stimulus = np.argmax(np.bincount(last_stimulus))
            
            records.append({
                "emg": padded_segment,
                "stimulus": int(dominant_stimulus)
            })

    return pd.DataFrame(records)

class Nina1Dataset(Dataset):
    def __init__(self, file_path, window_size=20, step_size=20):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)

        self.window_size = window_size
        self.step_size = step_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emg_data = np.array(row["emg"])
        label = row["stimulus"]

        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(-1, 10)

        fixed_length = emg_data.shape[0]
        windows = []
        
        for start in range(0, fixed_length - self.window_size + 1, self.step_size):
            window = emg_data[start:start + self.window_size, :]
            windows.append(window)

        input_data = np.stack(windows)  # (num_windows, window_size, 10)

        return (
            torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

class EMGStreamClassifier:
    def __init__(self, model, window_size=20, step_size=20, 
                 smoothing_window=5, stable_predictions=3, device='cuda'):
        self.model = model.to(device)
        self.window_size = window_size
        self.step_size = step_size
        self.smoothing_window = smoothing_window
        self.stable_predictions = stable_predictions
        self.device = device
        self.buffer = np.zeros((0, 10))
        
        # История для сглаживания
        self.pred_history = deque(maxlen=smoothing_window)
        
        # История для устойчивых предсказаний
        self.stable_history = deque(maxlen=stable_predictions)
        self.current_class = None
        
    def _get_window_prediction(self, window):
        """Предсказание для одного окна"""
        with torch.no_grad():
            inputs = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)
            return torch.argmax(outputs).item()
    
    def _update_stable_prediction(self, new_pred):
        """Обновление истории устойчивых предсказаний"""
        if len(self.stable_history) > 0 and new_pred == self.stable_history[-1]:
            self.stable_history.append(new_pred)
        else:
            self.stable_history.clear()
            self.stable_history.append(new_pred)
            
        # Если набрали нужное количество одинаковых предсказаний
        if len(self.stable_history) == self.stable_predictions:
            return new_pred
        return None

    def add_data(self, new_emg):
        """Добавляет новые EMG-данные в буфер"""
        self.buffer = np.vstack((self.buffer, new_emg))

    def process(self):
        """Обрабатывает буфер и возвращает итоговый класс"""
        if len(self.buffer) < self.window_size:
            return None

        # Обрабатываем все возможные окна в буфере
        results = []
        start = 0
        while start + self.window_size <= len(self.buffer):
            window = self.buffer[start:start + self.window_size]
            pred = self._get_window_prediction(window)
            results.append(pred)
            start += self.step_size

        # Если есть предсказания
        if results:
            # Выбираем наиболее частое предсказание для этого набора окон
            final_pred = max(set(results), key=results.count)
            
            # Добавляем в историю для сглаживания
            self.pred_history.append(final_pred)
            
            # Применяем сглаживание - выбираем наиболее частый класс в истории
            if len(self.pred_history) == self.smoothing_window:
                smoothed_pred = max(set(self.pred_history), key=self.pred_history.count)
                
                # Проверяем устойчивость предсказания
                stable_pred = self._update_stable_prediction(smoothed_pred)
                
                # Обновляем текущий класс если есть устойчивое предсказание
                if stable_pred is not None:
                    self.current_class = stable_pred
                
                # Очищаем буфер 
                self.buffer = self.buffer[-(self.step_size + self.window_size - 1):]
                
                return self.current_class

        return None

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        import os
        print("train_path:", self.train_path)
        print("Files in train_path:", os.listdir(self.train_path))
        
        train_df = load_mat_files_from_folder(self.train_path)
        print("test_path:", self.test_path)
        print("Files in test_path:", os.listdir(self.test_path))
        test_df = load_mat_files_from_folder(self.test_path)

        train_pkl = os.path.join(os.path.dirname(self.train_path), "train_temp.pkl")
        test_pkl = os.path.join(os.path.dirname(self.test_path), "test_temp.pkl")

        train_df.to_pickle(train_pkl)
        test_df.to_pickle(test_pkl)

        full_dataset = Nina1Dataset(train_pkl)
        test_dataset = Nina1Dataset(test_pkl)

        total_indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(total_indices, test_size=0.3, random_state=21)

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = test_dataset

    def teardown(self, stage: str) -> None:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == "__main__":
    dm = MyDataModule(
        train_path="D:/VScode/prostheses/data/train_data",     
        test_path="D:/VScode/prostheses/data/test_data",      
        batch_size=32,
        num_workers=0  
    )
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    print("Steps per epoch:", len(train_loader))
    print("Batch size:", train_loader.batch_size)
    print("Train dataset size:", len(train_loader.dataset))
    batch = next(iter(train_loader))
    print("Batch shape:", batch[0].shape)