"""Модуль, реализующий модель обучения для классификации ЭМГ сигналов.

Этот модуль содержит класс EMGHandNet_classifier, который расширяет LightningModule
для обучения нейронной сети. Включает в себя методы для обучения, валидации,
вычисления гессиана и настройки оптимизатора.
"""

import logging
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_modules.core_arch import EMGHandNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EMGHandNet_classifier(pl.LightningModule):
    """Классификатор ЭМГ сигналов на основе PyTorch Lightning.

    Этот класс реализует полный цикл обучения модели, включая:
    - Настройку архитектуры и параметров обучения
    - Методы для тренировки и валидации
    - Вычисление гессиана для анализа обучения
    - Настройку оптимизатора и планировщика скорости обучения
    - Работу с матрицей вероятностей для каждого окна
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        hessian_freq: int = 10,
        window_size: int = 500,  # Размер входного окна
        sliding_window_size: int = 25,  # Размер скользящего окна
    ):
        """Инициализация модели.

        Args:
            learning_rate: Скорость обучения
            weight_decay: Коэффициент регуляризации
            hessian_freq: Частота вычисления гессиана
            window_size: Размер входного окна (количество сэмплов)
            sliding_window_size: Размер скользящего окна для LSTM
        """
        super().__init__()
        self.save_hyperparameters()

        # Инициализация модели с гибкими параметрами размеров
        self.model = EMGHandNet(
            window_size=window_size, sliding_window_size=sliding_window_size
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hessian_freq = hessian_freq

        # Метрики для отслеживания точности
        self.train_acc = pl.metrics.Accuracy()  # Общая точность
        self.val_acc = pl.metrics.Accuracy()  # Точность на валидации
        self.test_acc = pl.metrics.Accuracy()  # Точность на тесте

        # Для гессиана
        self.hessian_condition_number = None
        self.hessian_trace = None

        # История матриц вероятностей для анализа
        # Каждый элемент - матрица размером [sliding_window_size, num_classes]
        self.probability_matrix_history = []

    def compute_hessian(self, loss: torch.Tensor) -> Tuple[float, float]:
        """Вычисляет гессиан функции потерь.

        Args:
            loss: Значение функции потерь

        Returns:
            Кортеж (число обусловленности, след) гессиана
        """
        try:
            # Градиент
            grad = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            grad = torch.cat([g.flatten() for g in grad])

            # Гессиан
            hessian_matrix = torch.zeros((len(grad), len(grad)), device=self.device)
            for i in range(len(grad)):
                hessian_matrix[i] = torch.autograd.grad(
                    grad[i], self.parameters(), create_graph=False, retain_graph=True
                )
                hessian_matrix[i] = torch.cat([h.flatten() for h in hessian_matrix[i]])

            # Собственные значения
            eigenvalues = torch.linalg.eigvals(hessian_matrix).real
            condition_number = torch.max(torch.abs(eigenvalues)) / (
                torch.min(torch.abs(eigenvalues)) + 1e-8
            )
            trace = torch.trace(hessian_matrix)

            return condition_number.item(), trace.item()

        except (RuntimeError, ValueError) as e:
            logging.error(f"Ошибка при вычислении гессиана: {e}")
            return 0.0, 0.0

    def compute_window_accuracy(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Вычисляет точность классификации для каждого окна.

        Args:
            probs: Матрица вероятностей [sliding_window_size, num_classes]
                  Каждая строка - вероятности классов для одного окна
            targets: Целевые метки [batch_size]
                    Одна метка для всего батча

        Returns:
            Средняя точность по всем окнам (от 0 до 1)
        """
        # Получаем предсказания для каждого окна
        predictions = probs.argmax(dim=1)  # [sliding_window_size]
        # Сравниваем с целевой меткой и усредняем
        correct = (predictions == targets).float().mean()
        return correct.item()

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

        # Гессиан (редко)
        if batch_idx % self.hessian_freq == 0:
            condition_number, trace = self.compute_hessian(loss)
            self.log("hessian_condition", condition_number)
            self.log("hessian_trace", trace)

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

        # Гессиан (редко)
        if batch_idx % self.hessian_freq == 0:
            condition_number, trace = self.compute_hessian(loss)
            self.log("val_hessian_condition", condition_number)
            self.log("val_hessian_trace", trace)

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
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
