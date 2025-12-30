from pathlib import Path

import lightning as L  # noqa: N812
import torch
from lightning.pytorch.loggers import MLFlowLogger

from image_generation_evaluation.data import AFHQv2DataModule
from image_generation_evaluation.models.wgan_gp.model import Critic, Generator
from image_generation_evaluation.models.wgan_gp.module import WGANGPModule


def train_wgan_gp():
    """Train the WGAN-GP model on the AFHQv2 dataset."""

    # TODO: use hydra config for managing hyperparameters

    L.seed_everything(42, workers=True)

    datamodule = AFHQv2DataModule(
        root_path=Path("data/AFHQv2"),
        image_size=64,
        batch_size=128,
    )

    model = WGANGPModule(
        generator=Generator(latent_vector_size=128),
        critic=Critic(),
        latent_dim=128,
        lr=1e-4,
        beta1=0.0,
        beta2=0.9,
        lambda_gp=10.0,
        n_critic=5,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="WGAN-GP-AFHQv2",
        tracking_uri="http://127.0.0.1:8080",
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=1,
        logger=mlflow_logger,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=True,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_wgan_gp()
