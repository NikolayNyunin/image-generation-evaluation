import fire

from image_generation_evaluation.models.wgan_gp.train import train_wgan_gp  # noqa: F401

if __name__ == "__main__":
    fire.Fire()
