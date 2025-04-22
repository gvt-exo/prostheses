from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Add L2 regularization loss
        l2_lambda = 0.01
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg

        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self):
        pass

    def predict_step(self, batch: Any) -> Any:
        data, logits = batch
        preds = self(data)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        return acc.item()

<<<<<<< HEAD
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                max_lr=5e-4,
                steps_per_epoch=100,
                epochs=self.config["training"]["num_epochs"],
                pct_start=0.2,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=1e2,
                verbose=True,
            ),
            "monitor": "val_acc",
            "interval": "step",
=======
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-5,  # Lower base LR
            weight_decay=0.1,  # Increased weight_decay further
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-4,  # Decreased max_lr again
            total_steps=self.trainer.estimated_stepping_batches,  # Correct way for Lightning
            pct_start=0.4,  # Keep long warmup
            div_factor=20.0,  # Adjusted div_factor (max_lr / base_lr = 2e-4 / 1e-5 = 20)
            final_div_factor=1e3,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
>>>>>>> 138de80 (попытка повысить точность)
        }
