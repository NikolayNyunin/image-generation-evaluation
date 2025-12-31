import fire
from hydra import compose, initialize
from omegaconf import DictConfig

from image_generation_evaluation.train import train as train_raw


def compose_cfg(overrides: list[str]) -> DictConfig:
    """Compose the hydra config with overrides using Compose API.

    Args:
        overrides: List of strings for overriding Hydra hyperparameters.
    """

    with initialize(config_path="configs", version_base=None):
        return compose(config_name="config", overrides=overrides)


def train(*overrides: str) -> None:
    """Initialize model training.

    Args:
        overrides: Strings "key=value" passed to Hydra for overriding default hyperparameters.
    """

    train_raw(compose_cfg(list(overrides)))


def main() -> None:
    """The main entrypoint of the project."""

    fire.Fire(
        {
            "train": train,
        }
    )


if __name__ == "__main__":
    main()
