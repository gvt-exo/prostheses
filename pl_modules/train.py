import pytorch_lightning as pl
from core_arch import EMGHandNet
from data import MyDataModule
from model import EMGHandNet_classifier
from pytorch_lightning.loggers import MLFlowLogger


mlflow_logger = MLFlowLogger(
    experiment_name="my_experiment", tracking_uri="file:./mlruns"
)  # Локально
trainer = pl.Trainer(logger=mlflow_logger)


def main():
    pl.seed_everything(42)

    dm = MyDataModule(
        train_path=r"C:\projects\prostheses\data\ninaprodb1train.pkl",
        test_path=r"C:\projects\prostheses\data\ninaprodb1test.pkl",
        batch_size=32,
        num_workers=4,
    )

    model = EMGHandNet_classifier(EMGHandNet(num_classes=52), lr=1e-4)
    trainer = pl.Trainer(
        max_epochs=10,
        logger=mlflow_logger,
        # log_every_n_steps=1,  # to resolve warnings
        accelerator="gpu",
        devices="auto",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
