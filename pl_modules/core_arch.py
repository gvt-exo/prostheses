import torch.nn as nn


class EMGHandNet(nn.Module):
    """Тут описывается основное ядро архитектуры, именно её слои и методы forward.
    Далее это ядро передается в файл модели обучения model.py, где указываются
    парамеры уже обучения.
    """

    def __init__(self, num_classes=36):
        super(EMGHandNet, self).__init__()

        # Модифицированная CNN часть (уменьшаем размеры)
        self.cnn = nn.Sequential(
            nn.Conv1d(10, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Dropout(0.3),
        )

        # Bi-LSTM часть (уменьшаем размеры)
        self.bilstm = nn.LSTM(
            input_size=512,  # 128*4
            hidden_size=128,
            num_layers=1,  # уменьшаем количество слоев
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        # Полносвязные слои (уменьшаем размеры)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 (bidirectional)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Обработка временных окон
        x = x.view(-1, 20, 10)
        x = x.permute(0, 2, 1)

        # CNN обработка
        x = self.cnn(x)

        # Подготовка к LSTM
        x = x.view(batch_size, 25, -1)

        # Bi-LSTM
        x, _ = self.bilstm(x)
        x = x[:, -1, :]

        # Классификация
        return self.fc(x)
