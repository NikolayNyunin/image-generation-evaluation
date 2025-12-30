from pathlib import Path

import lightning as L  # noqa: N812
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class AFHQv2DataModule(L.LightningDataModule):
    """LightningDataModule for the AFHQv2 dataset."""

    def __init__(  # noqa: PLR0913
        self,
        root_path: str | Path = Path("data/AFHQv2"),
        image_size: int = 128,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
    ) -> None:
        """Initialize AFHQv2DataModule.

        Args:
            root_path: Path to the root directory of the AFHQv2 dataset.
            image_size: Size to which the images are resized. Defaults to 128.
            batch_size: Batch size. Defaults to 64.
            num_workers: Number of workers used in data loading. Defaults to 4.
            pin_memory: Whether to use pin_memory. Defaults to True.
            shuffle: Whether to shuffle the data. Defaults to True.
        """
        super().__init__()

        self.root_path = Path(root_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.dataset: ImageFolder | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create dataset. Called once per process."""

        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset = ImageFolder(root=self.root_path, transform=transform)

    def train_dataloader(self) -> DataLoader:
        """Dataloader for training the models."""

        if self.dataset is None:
            raise RuntimeError("DataModule.setup() must be called before train_dataloader().")

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True,
        )
