import torch
from torch import nn
from torch.nn import functional as F, init

from .types_ import *


def kaiming_init(m):
    """Applies Kaiming Initialization to Conv2D and Linear layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


class SiFall(nn.Module):

    def __init__(
            self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
    ) -> None:
        super(SiFall, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(h_dim, affine=True),
                    nn.LeakyReLU(),

                    nn.Conv2d(h_dim, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(h_dim, affine=True),
                    nn.LeakyReLU(),

                    nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.MaxUnpool2d(kernel_size=2, stride=2),
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(hidden_dims[i + 1], affine=True),
                    nn.LeakyReLU(),

                    nn.ConvTranspose2d(hidden_dims[i + 1], hidden_dims[i + 1],
                                       kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(hidden_dims[i + 1], affine=True),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_dims[-1], affine=True),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_dims[-1], affine=True),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.apply(kaiming_init)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        pooling_indices = []

        for layer in self.encoder:
            input, indices = layer(input)
            pooling_indices.append(indices.to(dtype=torch.int64))

        input = self.avg_pool(input)
        input = torch.flatten(input, start_dim=1)

        # Compute latent variables
        mu = self.fc_mu(input)
        log_var = self.fc_var(input)

        return [mu, log_var, pooling_indices]

    def decode(self, z: Tensor, pooling_indices: List[Tensor]) -> Tensor:
        """
        Maps the given latent codes onto the reconstructed image space.

        :param z: (Tensor) [B x D]
        :param pooling_indices: (List[Tensor]) Pooling indices from encoder
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)

        pooling_indices = list(reversed(pooling_indices))
        for layer, indices in zip(self.decoder, pooling_indices):
            result = layer[0](result, indices)
            result = layer[1:](result)

        result = self.final_layer[0](result, pooling_indices[-1])
        result = self.final_layer[1:](result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, pooling_indices = self.encode(input)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z, pooling_indices)
        return [reconstructed, input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(
            self, num_samples: int, current_device: int, **kwargs
    ) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        mu = torch.zeros(num_samples, self.latent_dim).cuda()
        log_var = torch.ones(num_samples, self.latent_dim).cuda()
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).cuda()
        z = mu + std * eps

        expected_shapes = [
            (num_samples, 512, 2, 2),
            (num_samples, 256, 4, 4),
            (num_samples, 128, 8, 8),
            (num_samples, 64, 16, 16),
            (num_samples, 32, 32, 32)
        ]

        dummy_pooling_indices = []
        for shape in reversed(expected_shapes):
            indices = torch.zeros(shape, dtype=torch.int64).cuda()
            dummy_pooling_indices.append(indices)

        samples = self.decode(z, dummy_pooling_indices)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
