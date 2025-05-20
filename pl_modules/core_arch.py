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
        window_size: int = 500,  # Размер входного окна (количество сэмплов ЭМГ)
        num_channels: int = 10,  # Количество каналов ЭМГ
        cnn_filters=None,
        kernel_sizes=None,
        fc_units=None,
        num_classes=17,
        sliding_window_size: int = 25,  # Размер скользящего окна для LSTM
    ):
        """Инициализация модели.

        Args:
            window_size: Размер входного окна (количество сэмплов)
            num_channels: Количество каналов ЭМГ
            cnn_filters: Список количества фильтров для CNN слоев
            kernel_sizes: Список размеров ядер для CNN слоев
            fc_units: Список размеров полносвязных слоев
            num_classes: Количество классов для классификации
            sliding_window_size: Размер скользящего окна для LSTM
        """
        super().__init__()
        # Сохраняем параметры для гибкой настройки размеров
        self.window_size = window_size
        self.num_channels = num_channels
        self.sliding_window_size = sliding_window_size
        self.num_classes = num_classes

        # Значения по умолчанию
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

        # Вычисляем размерность после CNN слоев с учетом MaxPool2d
        # Каждый MaxPool2d уменьшает размерность в 2 раза
        cnn_output_height = window_size // (2 ** len(cnn_filters))  # Из-за MaxPool2d
        cnn_output_width = num_channels // (2 ** len(cnn_filters))

        # Сверточные слои
        self.cnn_layers = nn.ModuleList()
        in_channels = num_channels

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
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            in_channels = filters

        # LSTM слой с динамическим размером входа
        # Размер входа зависит от выходной размерности CNN
        lstm_input_size = cnn_filters[-1] * cnn_output_height * cnn_output_width
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,  # Динамический размер входа
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Полносвязные слои
        self.fc_layers = nn.ModuleList()
        lstm_output_size = 256  # 128 * 2 (bidirectional)

        for units in fc_units:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(lstm_output_size, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
            )
            lstm_output_size = units

        # Выходной слой с вероятностями
        self.fc_out = nn.Linear(lstm_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Для получения вероятностей классов

        # Буфер для хранения матрицы вероятностей
        # Размер: [sliding_window_size, num_classes]
        # Каждая строка - вероятности классов для одного окна
        self.register_buffer(
            "probability_matrix", torch.zeros((sliding_window_size, num_classes))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход модели.

        Args:
            x: Входные данные в одном из форматов:
               - [batch_size, window_size, num_channels] - одиночное окно
               - [batch_size, sliding_window_size, window_size, num_channels] - набор окон

        Returns:
            Кортеж (предсказания, матрица вероятностей):
            - предсказания: [batch_size, num_classes] - финальные предсказания
            - матрица вероятностей: [sliding_window_size, num_classes] - вероятности для каждого окна
        """
        batch_size = x.size(0)

        # Если вход имеет размер [batch_size, window_size, num_channels],
        # добавляем размерность для sliding_window
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Добавляем размерность для sliding_window

        # Преобразуем вход в формат для CNN
        # Из [batch_size, sliding_window_size, window_size, num_channels]
        # в [batch_size * sliding_window_size, num_channels, window_size, 1]
        x = x.reshape(-1, self.num_channels, self.window_size, 1)

        # Применяем сверточные слои
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)

        # Подготовка данных для LSTM
        # Преобразуем обратно в [batch_size, sliding_window_size, features]
        x = x.reshape(batch_size, self.sliding_window_size, -1)

        # Применяем LSTM
        lstm_out, _ = self.lstm(x)

        # Получаем предсказания для каждого окна
        window_predictions = []
        for i in range(self.sliding_window_size):
            # Берем выход LSTM для текущего окна
            window_x = lstm_out[:, i, :]

            # Применяем полносвязные слои
            for fc_layer in self.fc_layers:
                window_x = fc_layer(window_x)

            # Получаем логиты и вероятности для окна
            window_logits = self.fc_out(window_x)
            window_probs = self.softmax(window_logits)  # [batch_size, num_classes]
            window_predictions.append(window_probs)

            # Обновляем матрицу вероятностей для текущего окна
            # Усредняем по батчу для получения общих вероятностей
            self.probability_matrix[i] = window_probs.mean(dim=0)

        # Усредняем предсказания по всем окнам для финального результата
        final_predictions = torch.stack(window_predictions).mean(dim=0)

        return final_predictions, self.probability_matrix
