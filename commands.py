import fire

from image_generation_evaluation.models.wgan_gp.train import train_wgan_gp


def main() -> None:
    """The main entrypoint of the project."""

    fire.Fire(
        {
            "train-wgan-gp": train_wgan_gp,
        }
    )


if __name__ == "__main__":
    main()
