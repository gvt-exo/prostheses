from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


class EMGHandNet_classifier(pl.LightningModule):
    """Тут производится полная настройка модели обучения.
    Указывается архитектура, которую будем обучать, настраиваются шаги и оптимизаторы.
    Так же добавляются точки логгирования, индивидуально для каждого типа шага
    (валидационный или тренировочный)."""

    def __init__(self, model, lr, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.config = config
        self.optim_conf = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch: Any):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"loss": loss, "val_acc": acc}

    def test_step(self):
        pass

    def predict_step(self, batch: Any) -> Any:
        data, logits = batch
        preds = self(data)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        return acc.item()

    def on_fit_start(self):
        # Получаем количество шагов на эпоху и общее число шагов
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = steps_per_epoch * self.trainer.max_epochs

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.25,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e2,
            verbose=True,
        )
        self.optim_conf = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # важно!
            },
        }

    def configure_optimizers(self):
        if self.optim_conf is None:
            # Запасной вариант, если on_fit_start еще не был вызван
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer
        return self.optim_conf