import hydra
import pytorch_lightning as pl
import torch
from core_arch import EMGHandNet
from data import MyDataModule
from model import EMGHandNet_classifier
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger


mlflow_logger = MLFlowLogger(
    experiment_name="my_experiment", tracking_uri="file:./mlruns"
)
trainer = pl.Trainer(logger=mlflow_logger)

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)

    dm = MyDataModule(
        train_path=config["data_loading"]["train_data_path"],
        test_path=config["data_loading"]["test_data_path"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    model = EMGHandNet_classifier(
        EMGHandNet(num_classes=config["model"]["num_classes"]),
        lr=config["training"]["lr"],
        config=config,
    )

    loggers = [
        pl.loggers.WandbLogger(
            project=config["logging"]["project"],
            name=config["logging"]["name"],
            save_dir=config["logging"]["save_dir"],
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        # pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        logger=loggers,
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
