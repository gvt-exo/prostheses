"""Модуль для обучения модели классификации ЭМГ сигналов.

Этот модуль содержит функции для настройки и запуска процесса обучения модели,
включая загрузку конфигурации, подготовку данных и запуск обучения.
"""

# Настройка логирования
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pl_modules.data import MyDataModule
from pl_modules.model import EMGHandNet_classifier
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(config: DictConfig) -> None:
    """Запускает процесс обучения модели.

    Args:
        config: Конфигурация обучения, загружаемая через Hydra
    """
    # Создаем директорию для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Инициализируем логгер
    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name="emg_classifier",
        version=None,
    )

    # Создаем модель
    classifier = EMGHandNet_classifier(
        learning_rate=config.training.lr,
        weight_decay=config.training.weight_decay,
        window_size=config.data_loading.window_size,
        sliding_window_size=config.data_loading.sliding_window_size,
        num_classes=config.model.num_classes,
    )

    # Настраиваем колбэки
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,  # Уменьшаем patience для более быстрой остановки
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=log_dir / "emg_classifier",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
    ]

    # Создаем тренер
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator="auto",  # Автоматически выбираем доступное устройство
        devices=1,  # Используем одно устройство
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # Инициализируем модуль данных
    data_module = MyDataModule(
        data_root=config.data_loading.data_root,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        window_size=config.data_loading.window_size,
        sliding_window_size=config.data_loading.sliding_window_size,
        step_size=config.data_loading.step_size,
    )

    # Запускаем обучение
    trainer.fit(
        model=classifier,
        datamodule=data_module,
    )

    # Сохраняем конфигурацию
    OmegaConf.save(
        config=config,
        f=log_dir / "emg_classifier" / "config.yaml",
    )


if __name__ == "__main__":
    train()
