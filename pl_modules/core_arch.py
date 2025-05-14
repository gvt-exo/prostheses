"""Модуль, содержащий основную архитектуру нейронной сети для классификации ЭМГ.

Этот модуль определяет класс EMGHandNet, который реализует гибридную архитектуру
CNN-LSTM для обработки и классификации электромиографических сигналов.
"""

import torch
import torch.nn as nn


class EMGHandNet(nn.Module):
    """Архитектура нейронной сети для классификации ЭМГ сигналов.

    Состоит из последовательности сверточных слоев, LSTM и полносвязных слоев.
    """

    def __init__(
        self,
        cnn_filters=None,
        kernel_sizes=None,
        fc_units=None,
        num_classes=17,
    ):
        """Инициализация модели.

        Args:
            cnn_filters: Список количества фильтров для CNN слоев
            kernel_sizes: Список размеров ядер для CNN слоев
            fc_units: Список размеров полносвязных слоев
            num_classes: Количество классов для классификации
        """
        super().__init__()

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

        # Сверточные слои
        self.cnn_layers = nn.ModuleList()
        in_channels = 10  # Начальное количество каналов

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

        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=cnn_filters[2]
            * 4,  # Применяется для соответствия размерности после CNN
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

        # Выходной слой
        self.fc_out = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели.

        Args:
            x: Входные данные [batch_size, 25, 20, 10]

        Returns:
            Выход модели [batch_size, num_classes]
        """
        # Применяем сверточные слои
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)

        # Подготовка данных для LSTM
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, x.size(1) * x.size(2))

        # Применяем LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Берем последний выход LSTM

        # Применяем полносвязные слои
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Выходной слой
        x = self.fc_out(x)

        return x
