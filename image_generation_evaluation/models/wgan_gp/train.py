from pathlib import Path

import lightning as L  # noqa: N812
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from image_generation_evaluation.data import AFHQv2DataModule
from image_generation_evaluation.models.wgan_gp.model import Critic, Generator
from image_generation_evaluation.models.wgan_gp.module import WGANGPModule


def train_wgan_gp(cfg: DictConfig) -> None:
    """Train the WGAN-GP model on the AFHQv2 dataset."""

    L.seed_everything(cfg.system.global_seed, workers=cfg.system.seed_workers)

    datamodule = AFHQv2DataModule(
        root_path=Path(cfg.data.root_path),
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=cfg.data.shuffle,
    )

    model = WGANGPModule(
        generator=Generator(cfg.model.latent_dim),
        critic=Critic(),
        latent_dim=cfg.model.latent_dim,
        lr=cfg.model.lr,
        beta1=cfg.model.beta1,
        beta2=cfg.model.beta2,
        lambda_gp=cfg.model.lambda_gp,
        n_critic=cfg.model.n_critic,
        n_example=cfg.model.n_example,
    )

    if cfg.logging.name == "mlflow":
        logger = MLFlowLogger(
            experiment_name=cfg.logging.mlflow_experiment_name,
            tracking_uri=cfg.logging.mlflow_tracking_uri,
        )
        logger.log_hyperparams(cfg.data)
        logger.log_hyperparams(cfg.train)
    else:
        raise ValueError(f"Logger name '{cfg.logging.name}' not recognized")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.system.devices,
        max_epochs=cfg.train.max_epochs,
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_checkpointing=cfg.train.enable_checkpointing,
        enable_model_summary=cfg.train.enable_model_summary,
        deterministic=cfg.system.deterministic,
    )

    trainer.fit(model, datamodule=datamodule)
