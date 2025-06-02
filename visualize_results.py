from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from omegaconf import DictConfig
from pl_modules.data import MyDataModule
from pl_modules.model import EMGHandNet_classifier


def plot_accuracy_history(trainer):
    """Plot model accuracy metrics."""
    plt.figure(figsize=(10, 6))

    # Get metrics from trainer
    test_acc = trainer.callback_metrics["test_acc"].item()
    test_window_acc = trainer.callback_metrics["test_window_acc"].item()

    # Plot bars
    plt.bar(["Overall Accuracy", "Window Accuracy"], [test_acc, test_window_acc])
    plt.title("Model Accuracy Metrics")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    # Add value labels on top of bars
    for i, v in enumerate([test_acc, test_window_acc]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.savefig("accuracy_plot.png")
    plt.close()


def plot_probability_matrix(model, dataloader):
    """Plot probability matrix for a single movement."""
    model.eval()
    with torch.no_grad():
        # Get a single batch
        batch = next(iter(dataloader))
        x, y = batch

        # Get predictions
        _, prob_matrix = model(x)

        # Convert to numpy for plotting
        prob_matrix = prob_matrix.cpu().numpy()

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            prob_matrix.T,
            cmap="YlOrRd",
            xticklabels=10,
            yticklabels=range(prob_matrix.shape[1]),
            cbar_kws={"label": "Probability"},
        )

        plt.title("Probability Matrix for Movement")
        plt.xlabel("Time Steps")
        plt.ylabel("Class")

        plt.savefig("probability_matrix.png")
        plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Get the project root directory
    project_root = Path(__file__).resolve().parent

    # Initialize data module with absolute path
    data_module = MyDataModule(
        data_root=str(project_root / "data"),
        batch_size=cfg.data.batch_size,
        window_size=cfg.model.window_size,
        sliding_window_size=cfg.model.sliding_window_size,
    )

    # Initialize model
    model = EMGHandNet_classifier(
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        window_size=cfg.model.window_size,
        sliding_window_size=cfg.model.sliding_window_size,
        num_classes=cfg.model.num_classes,
    )

    # Load the trained model
    checkpoint_path = project_root / cfg.model.checkpoint_path
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        # Load state dict directly without prefix handling
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,  # We only need one epoch for visualization
        accelerator="auto",
        devices=1,
    )

    # Get accuracy metrics
    trainer.test(model, datamodule=data_module)

    # Create visualizations
    plot_accuracy_history(trainer)
    plot_probability_matrix(model, data_module.test_dataloader())


if __name__ == "__main__":
    main()
