from omegaconf import DictConfig

from image_generation_evaluation.models.wgan_gp.train import train_wgan_gp

MODELS = {
    "WGAN-GP": train_wgan_gp,
}


def train(cfg: DictConfig) -> None:
    """Launch training of the selected model.

    Args:
        cfg: Hydra config for training.
    """

    if cfg.model.name in MODELS:
        MODELS[cfg.model.name](cfg)
    else:
        raise ValueError(f"Model name '{cfg.model.name}' not recognized")
