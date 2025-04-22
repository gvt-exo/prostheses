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
        cnn_filters=None,  # Уменьшаем размеры фильтров
        kernel_sizes=None,  # Размеры ядер сверточных слоев
        fc_units=None,  # Уменьшаем размер полносвязного слоя
        lstm_hidden_size=128,  # Уменьшаем размер LSTM
        lstm_layers=2,  # Увеличиваем количество слоев
        dropout=0.6,  # Увеличиваем dropout еще
    ):
        super(EMGHandNet, self).__init__()

        if cnn_filters is None:
            cnn_filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if fc_units is None:
            fc_units = [256]

        # CNN часть с усиленной регуляризацией
        self.conv1 = nn.Conv1d(
            input_channels, cnn_filters[0], kernel_size=kernel_sizes[0], padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            cnn_filters[0], cnn_filters[1], kernel_size=kernel_sizes[1], padding=1
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(
            cnn_filters[1], cnn_filters[2], kernel_size=kernel_sizes[2], padding=1
        )
        self.bn3 = nn.BatchNorm1d(cnn_filters[2])
        self.dropout3 = nn.Dropout(dropout)

        # Bi-LSTM с регуляризацией
        self.lstm = nn.LSTM(
            input_size=cnn_filters[2] * 20,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Полносвязные слои с регуляризацией
        self.fc1 = nn.Linear(lstm_hidden_size * 2, fc_units[0])
        self.bn_fc1 = nn.BatchNorm1d(fc_units[0])
        self.dropout_fc1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(fc_units[0], num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Обработка временных окон
        x = x.view(-1, 20, 10)  # [batch*25, 20, 10]
        x = x.permute(0, 2, 1)  # [batch*25, 10, 20]

        # CNN обработка
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.tanh(x)  # Используем Tanh для стабильности градиентов
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.tanh(x)
        x = self.dropout3(x)

        # Подготовка для LSTM
        x = x.view(batch_size, 25, -1)  # [batch, 25, 1024]

        # Bi-LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Берем последний временной шаг

        # Полносвязные слои
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.tanh(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)

        return x
