import torch
from torch import nn


class Critic(nn.Module):
    """`Critic` network for predicting the "realness" scores for images in WGAN-GP architecture."""

    def __init__(self):
        super().__init__()

        channel_counts = (3, 32, 64, 128, 256)
        layers = []

        for i in range(len(channel_counts) - 1):
            in_channels, out_channels = channel_counts[i], channel_counts[i + 1]
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=256 * 4 * 4, out_features=1, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Critic.

        Args:
            x: Tensors of shape (n, 3, 64, 64) - n 64x64 RGB images.

        Returns:
            torch.Tensor: Scores for images.
        """
        return self.net(x).view(-1)


class Generator(nn.Module):
    """`Generator` network for generating images from latent vector space."""

    def __init__(self, latent_vector_size=256):
        """Initialize Generator.

        Args:
            latent_vector_size - Size of the latent vector space. Defaults to 256.
        """
        super().__init__()

        channel_counts = (latent_vector_size, 256, 128, 64, 32, 16)
        layers: list[nn.Module] = [nn.Unflatten(1, (latent_vector_size, 1, 1))]

        for i in range(len(channel_counts) - 1):
            in_channels, out_channels = channel_counts[i], channel_counts[i + 1]
            if i == 0:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                )
            else:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False
            )
        )
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.

        Args:
            x: Batch of latent vectors of shape (n, latent_vector_size).

        Returns:
            torch.Tensor: Generated images.
        """
        return self.net(x)
