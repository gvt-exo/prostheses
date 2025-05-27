"""Модуль, реализующий модель обучения для классификации ЭМГ сигналов.

Этот модуль содержит класс EMGHandNet_classifier, который расширяет LightningModule
для обучения нейронной сети. Включает в себя методы для обучения, валидации,
вычисления гессиана и настройки оптимизатора.
"""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_modules.core_arch import EMGHandNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy


class EMGHandNet_classifier(pl.LightningModule):
    """Классификатор ЭМГ сигналов на основе PyTorch Lightning."""

    # Этот класс реализует полный цикл обучения модели, включая:
    # - Настройку архитектуры и параметров обучения
    # - Методы для тренировки и валидации
    # - Вычисление гессиана для анализа обучения
    # - Настройку оптимизатора и планировщика скорости обучения
    # - Работу с матрицей вероятностей для каждого окна.

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        window_size: int = 500,  # Размер входного окна
        sliding_window_size: int = 25,  # Размер скользящего окна
        num_classes: int = 17,  # <-- добавляем
    ):
        """Инициализация модели.

        Args:
            learning_rate: Скорость обучения
            weight_decay: Коэффициент регуляризации
            window_size: Размер входного окна (количество сэмплов)
            sliding_window_size: Размер скользящего окна для LSTM
            num_classes: Количество классов
        """
        super().__init__()
        self.save_hyperparameters()

        # Инициализация модели с гибкими параметрами размеров
        self.model = EMGHandNet(
            window_size=window_size, sliding_window_size=sliding_window_size
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Метрики для отслеживания точности
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # История матриц вероятностей для анализа
        # Каждый элемент - матрица размером [window_steps, num_classes]
        self.probability_matrix_history = []

    def compute_window_accuracy(self, prob_matrix, targets):
        """Вычисляет точность по каждому окну (временной шаг)"""
        # prob_matrix: [window_steps, num_classes]
        # targets: [batch_size]
        # Для каждого окна берём argmax по классам
        predictions = prob_matrix.argmax(dim=1)  # [window_steps]
        # targets нужно привести к [window_steps] или [batch_size], если нужно сравнивать
        # Но правильнее — усреднять по batch, а не по окнам
        # Поэтому возвращаем среднюю точность по окнам
        correct = (predictions.unsqueeze(1) == targets.unsqueeze(0)).float().mean()
        return correct

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход модели.

        Args:
            x: Входные данные [batch_size, window_size, num_channels]
               или [batch_size, sliding_window_size, window_size, num_channels]

        Returns:
            Кортеж (предсказания, матрица вероятностей)
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Шаг обучения.

        Args:
            batch: Кортеж (данные, метки)
            batch_idx: Индекс батча

        Returns:
            Словарь с метриками, включая:
            - loss: значение функции потерь
            - train_acc: общая точность
            - train_window_acc: точность по окнам
        """
        x, y = batch
        # Получаем предсказания и матрицу вероятностей
        logits, prob_matrix = self(x)
        loss = F.cross_entropy(logits, y)

        # Обновление метрик
        self.train_acc(logits, y)  # Общая точность
        window_acc = self.compute_window_accuracy(prob_matrix, y)  # Точность по окнам

        # Сохраняем матрицу вероятностей для последующего анализа
        self.probability_matrix_history.append(prob_matrix.detach().cpu())

        # Логируем метрики
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_window_acc", window_acc, prog_bar=True)

        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Шаг валидации.

        Args:
            batch: Кортеж (данные, метки)
            batch_idx: Индекс батча

        Returns:
            Словарь с метриками
        """
        x, y = batch
        logits, prob_matrix = self(x)
        loss = F.cross_entropy(logits, y)

        # Обновление метрик
        self.val_acc(logits, y)
        window_acc = self.compute_window_accuracy(prob_matrix, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_window_acc", window_acc, prog_bar=True)

        return {"val_loss": loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Шаг тестирования.

        Args:
            batch: Кортеж (данные, метки)
            batch_idx: Индекс батча

        Returns:
            Словарь с метриками
        """
        x, y = batch
        logits, prob_matrix = self(x)
        loss = F.cross_entropy(logits, y)

        # Обновление метрик
        self.test_acc(logits, y)
        window_acc = self.compute_window_accuracy(prob_matrix, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_window_acc", window_acc, prog_bar=True)

        return {"test_loss": loss}

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Шаг предсказания.

        Args:
            batch: Входные данные
            batch_idx: Индекс батча

        Returns:
            Кортеж (предсказания, матрица вероятностей)
        """
        return self(batch)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Настройка оптимизатора и планировщика.

        Returns:
            Словарь с оптимизатором и планировщиком
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
