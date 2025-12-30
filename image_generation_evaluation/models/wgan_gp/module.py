from typing import Any

import lightning as L  # noqa: N812
import torch
from torch import nn


class WGANGPModule(L.LightningModule):
    """LightningModule for the WGAN-GP model."""

    def __init__(  # noqa: PLR0913
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int = 256,
        lr: float = 1e-4,
        b1: float = 0.0,
        b2: float = 0.9,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
    ) -> None:
        """Initialize the WGANGPModule.

        Args:
            generator: Generator network.
            critic: Critic network.
            latent_dim: Latent vector dimension. Defaults to 256.
            lr: Learning rate for Critic and Generator. Defaults to 1e-4.
            b1: beta1 for Adam optimization. Defaults to 0.0.
            b2: beta2 for Adam optimization. Defaults to 0.9.
            lambda_gp: Weight for GP loss. Defaults to 10.0.
            n_critic: Number of Critic optimization steps per Generator step. Defaults to 5.
        """
        super().__init__()

        self.generator = generator
        self.critic = critic

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.automatic_optimization = False

    def compute_gradient_penalty(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute the GP component of WGAN-GP loss."""

        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device, requires_grad=True)

        interpolated = epsilon * real_data + (1.0 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        prob_interpolated = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def training_step(self, batch: Any, batch_idx: int):
        """One step of training the model."""

        real_images, _ = batch
        batch_size = real_images.size(0)

        opt_g, opt_c = self.optimizers()

        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)

        c_real = self.critic(real_images).mean()
        c_fake = self.critic(fake_images.detach()).mean()

        gp = self.compute_gradient_penalty(real_images, fake_images.detach())

        c_loss = c_fake - c_real + gp

        opt_c.zero_grad()
        self.manual_backward(c_loss)
        opt_c.step()

        self.log("loss/critic", c_loss, prog_bar=True)
        self.log("critic/real", c_real, prog_bar=False)
        self.log("critic/fake", c_fake, prog_bar=False)
        self.log("critic/gp", gp, prog_bar=False)

        if (self.global_step + 1) % self.n_critic == 0:
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)

            g_loss = -self.critic(fake_images).mean()

            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()

            self.log("loss/generator", g_loss, prog_bar=True)

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers for Generator and Critic."""

        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )
        opt_c = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )

        return opt_g, opt_c
