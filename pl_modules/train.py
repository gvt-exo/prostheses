"""Модуль для обучения модели классификации ЭМГ сигналов.

Этот модуль содержит функции и классы для настройки и запуска
процесса обучения нейронной сети. Включает в себя конфигурацию
модели, логирование и сохранение результатов.
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from core_arch import EMGHandNet
from data import MyDataModule
from model import EMGHandNet_classifier
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="conf", config_name="config")
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
        name=config.model.name,
        version=None,
    )

    # Создаем модель
    model = EMGHandNet(
        num_classes=config.model.num_classes,
        input_channels=config.model.input_channels,
    )
    classifier = EMGHandNet_classifier(
        model=model,
        lr=config.training.lr,
        config=config,
    )

    # Настраиваем колбэки
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=log_dir / config.model.name,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
    ]

    # Создаем тренер
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # Инициализируем модуль данных
    data_module = MyDataModule(
        data_root=config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    # Запускаем обучение
    trainer.fit(
        model=classifier,
        datamodule=data_module,
    )

    # Сохраняем конфигурацию
    OmegaConf.save(
        config=config,
        f=log_dir / config.model.name / "config.yaml",
    )


if __name__ == "__main__":
    train()
