import torch.nn as nn


class EMGHandNet(nn.Module):
    def __init__(self, num_classes=52):
        super(EMGHandNet, self).__init__()

        # Модифицированная CNN часть
        self.cnn = nn.Sequential(
            nn.Conv1d(10, 64, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(4),  # Фиксирует выходной размер
            nn.Flatten(),
        )

        # Bi-LSTM часть
        self.bilstm = nn.LSTM(
            input_size=1024,  # 256*4
            hidden_size=200,
            num_layers=1,  # Уменьшаем количество слоев
            bidirectional=True,
            batch_first=True,
            # dropout=0.2,
        )

        # Полносвязные слои
        self.fc = nn.Sequential(
            nn.Linear(400, 512),  # 200*2 (bidirectional)
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Обработка временных окон
        x = x.view(-1, 20, 10)  # [batch*25, 20, 10]
        x = x.permute(0, 2, 1)  # [batch*25, 10, 20]

        # CNN обработка
        x = self.cnn(x)  # [batch*25, 256*4=1024]

        # Подготовка к LSTM
        x = x.view(batch_size, 25, -1)  # [batch, 25, 1024]

        # Bi-LSTM
        x, _ = self.bilstm(x)  # [batch, 25, 400]
        x = x[:, -1, :]  # Берем последний временной шаг

        # Классификация
        return self.fc(x)
