"""Модуль, содержащий основную архитектуру нейронной сети для классификации ЭМГ.

Этот модуль определяет класс EMGHandNet, который реализует гибридную архитектуру
CNN-LSTM для обработки и классификации электромиографических сигналов.
"""

from typing import Tuple

import torch
import torch.nn as nn


class EMGHandNet(nn.Module):
    """Архитектура нейронной сети для классификации ЭМГ сигналов.

    Состоит из последовательности сверточных слоев, LSTM и полносвязных слоев.
    Поддерживает гибкий размер входного окна.
    """

    def __init__(
        self,
        window_size: int = 200,
        num_channels: int = 10,
        cnn_filters=None,
        kernel_sizes=None,
        fc_units=None,
        num_classes=17,
        sliding_window_size: int = 50,
    ):
        """Инициализация модели.

        Args:
            window_size: Размер входного окна
            num_channels: Количество каналов ЭМГ
            cnn_filters: Список количества фильтров для каждого CNN слоя
            kernel_sizes: Список размеров ядер для каждого CNN слоя
            fc_units: Список размеров полносвязных слоев
            num_classes: Количество классов для классификации
            sliding_window_size: Размер скользящего окна
        """
        super().__init__()
        self.window_size = window_size
        self.num_channels = num_channels
        self.sliding_window_size = sliding_window_size
        self.num_classes = num_classes

        # Значения по умолчанию для архитектуры
        if cnn_filters is None:
            cnn_filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if fc_units is None:
            fc_units = [256, 128]

        # Проверка входных параметров
        if len(cnn_filters) != len(kernel_sizes):
            raise ValueError(
                "Количество фильтров должно совпадать с количеством размеров ядер"
            )

        # CNN слои
        self.cnn_layers = nn.ModuleList()
        in_channels = 1  # Изменено: входной канал теперь 1

        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, kernel_sizes)):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        filters,
                        kernel_size=(kernel_size, kernel_size),
                        padding="same",
                    ),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                )
            )
            in_channels = filters

        # Вычисляем размерность после CNN слоев
        cnn_output_height = window_size // (2 ** len(cnn_filters))  # 200 // 8 = 25
        cnn_output_width = num_channels  # 10
        self.window_steps = cnn_output_height  # Сохраняем для использования в forward и probability_matrix

        # LSTM слой
        lstm_input_size = cnn_filters[-1] * cnn_output_width  # 128 * 10 = 1280
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=256,  # Увеличено для лучшей производительности
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,  # Добавлен dropout для регуляризации
        )

        # Полносвязные слои
        self.fc_layers = nn.ModuleList()
        lstm_output_size = 512  # 256 * 2 (bidirectional)

        for units in fc_units:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(lstm_output_size, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU(),
                    nn.Dropout(0.4),  # Увеличен dropout
                )
            )
            lstm_output_size = units

        # Выходной слой
        self.fc_out = nn.Linear(lstm_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Буфер для матрицы вероятностей
        self.register_buffer(
            "probability_matrix", torch.zeros((self.window_steps, num_classes))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход модели.

        Args:
            x: Входные данные в формате [batch_size, window_size, num_channels]

        Returns:
            Кортеж (предсказания, матрица вероятностей):
            - предсказания: [batch_size, num_classes] - финальные предсказания
            - матрица вероятностей: [window_steps, num_classes] - вероятности для каждого окна
        """
        batch_size = x.size(0)

        # Добавляем размерность для каналов CNN
        x = x.unsqueeze(1)  # [batch_size, 1, window_size, num_channels]

        # Применяем сверточные слои
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)

        # Подготовка данных для LSTM
        # Преобразуем в [batch_size, window_steps, cnn_filters[-1] * num_channels]
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size, self.window_steps, x.size(1) * x.size(3)
        )

        # Применяем LSTM
        lstm_out, _ = self.lstm(x)

        # Получаем предсказания для каждого временного шага
        window_predictions = []
        for i in range(self.window_steps):
            window_x = lstm_out[:, i, :]

            for fc_layer in self.fc_layers:
                window_x = fc_layer(window_x)

            window_logits = self.fc_out(window_x)
            window_probs = self.softmax(window_logits)
            window_predictions.append(window_probs)

            # Обновляем матрицу вероятностей
            self.probability_matrix[i] = window_probs.mean(dim=0)

        # Усредняем предсказания по всем временным шагам
        final_predictions = torch.stack(window_predictions).mean(dim=0)

        return final_predictions, self.probability_matrix
