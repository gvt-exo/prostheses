from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm


class EMGHandNet_classifier(pl.LightningModule):
    def __init__(self, model, lr, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"loss": loss, "val_acc": acc}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data, logits = batch
        preds = self(data)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        return acc.item()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-3,
                steps_per_epoch=100,
                epochs=self.config["training"]["num_epochs"],
                pct_start=0.25,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
                verbose=True,
            ),
            "monitor": "loss",  # логгируете это в validation_step
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
