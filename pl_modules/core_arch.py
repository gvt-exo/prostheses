import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGHandNet(nn.Module):
    """Тут описывается основное ядро архитектуры, именно её слои и методы forward.
    Далее это ядро передается в файл модели обучения model.py, где указываются
    парамеры уже обучения.
    """

    def __init__(
        self,
        num_classes=17,
        input_channels=10,  # Количество каналов на входе (например, 10 для ЭМГ)
        cnn_filters=[64, 128, 256],  # Увеличиваем количество фильтров
        kernel_sizes=[3, 3, 3],  # Размеры ядер сверточных слоев
        lstm_hidden_size=200,  # Увеличиваем размер LSTM
        lstm_layers=1,  # Количество слоев в LSTM
        fc_units=[512],  # Количество нейронов в слоях после LSTM
        dropout=0.3,  # Уменьшаем dropout
    ):
        super(EMGHandNet, self).__init__()

        # Модифицированная CNN часть с настраиваемыми параметрами
        self.cnn = nn.Sequential(
            nn.Conv1d(
                input_channels,
                cnn_filters[0],
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                cnn_filters[0],
                cnn_filters[1],
                kernel_size=kernel_sizes[1],
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
            nn.BatchNorm1d(cnn_filters[1]),
            nn.Conv1d(
                cnn_filters[1],
                cnn_filters[2],
                kernel_size=kernel_sizes[2],
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
            nn.BatchNorm1d(cnn_filters[2]),
            nn.AdaptiveAvgPool1d(4),  # Фиксирует выходной размер
            nn.Flatten(),
        )

        # Bi-LSTM часть + настраиваемые параметры
        self.bilstm = nn.LSTM(
            input_size=cnn_filters[2]
            * 4,  # Применяется для соответствия размерности после CNN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,  # Количество слоев в LSTM
            bidirectional=True,
            batch_first=True,
        )

        # Полносвязные слои + настраиваемые параметры
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, fc_units[0]),  # *2 для bi-LSTM
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(fc_units[0], num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Обработка временных окон
        x = x.view(batch_size * 25, 20, 10)  # [batch*25, 20, 10]
        x = x.permute(0, 2, 1)  # [batch*25, 10, 20]

        # CNN обработка
        x = self.cnn(x)

        # Подготовка к LSTM
        x = x.view(batch_size, 25, -1)  # [batch, 25, 1024]

        # Bi-LSTM
        x, _ = self.bilstm(x)
        x = x[:, -1, :]  # Берем последний временной шаг

        # Полносвязные слои
        x = self.fc(x)

        return x
